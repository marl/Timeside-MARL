import argparse
import os
import sys

import numpy as np
import scipy.signal
import scipy.fftpack as fft

from librosa import cqt, magphase, note_to_hz, stft, resample, to_mono, load
from librosa import amplitude_to_db, get_duration, time_to_frames, power_to_db
from librosa.util import fix_length
from librosa.feature import melspectrogram, rms, tempogram
from librosa.decompose import hpss
from librosa.onset import onset_strength

from librosa.filters import get_window
from librosa.core.audio import resample
from librosa import util

from vggish import mel_features
from vggish import vggish_params

from multiprocessing import Pool


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('audio_directory', type=str,
                        help="The path to the directory of audio files.")

    parser.add_argument('output_directory', type=str,
                        help="Path to output directory.")

    parser.add_argument('cpus', type=int,
                        help="Number of processors")

    parser.add_argument('--overwrite', action='store_true', default=False)

    return parser.parse_args(args)


def _frames_helper(y, y_frames, n_fft, power=1.0):
    if y_frames is not None:
        D = frames_stft(y_frames=y_frames,
                        n_fft=n_fft)

        if power is not None:
            Sm = np.abs(D) ** power
            Sp = np.angle(D)
        else:
            return D
        y = None
    else:
        Sm = None
        Sp = None
        if y is None:
            raise Exception('y or frames must be defined.')
    return y, Sm, Sp


def frames_stft(y_frames, n_fft=2048, win_length=None, window='hann',
                dtype=np.complex64):
    """
    Adapted from librosa for frame input. NOTE: not centered anymore.
    """
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = util.pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(util.MAX_MEM_BLOCK / float(stft_matrix.shape[0] *
                                               stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]]

    return stft_matrix


def percussive_ratio(y=None, y_frames=None, n_fft=2048, hop_size=512, margin=1.0):
    """
    Compute ratio of percussive power to total power
    """

    # default for 22050
    if y is not None:
        D = stft(y, n_fft=n_fft, hop_length=hop_size)
    elif y_frames is not None:
        D = frames_stft(y_frames, n_fft=n_fft, hop_length=hop_size)
    H, P = hpss(D, margin=margin)

    Pm, Pp = magphase(P)
    S, phase = magphase(D)

    P_rms = rms(S=Pm)
    S_rms = rms(S=S)

    return amplitude_to_db(P_rms / S_rms), P_rms, S_rms


def melspec(y=None, y_frames=None, sr=22050, n_fft=2048,
            hop_size=512, n_mels=128, fmin=0.0):
    """
    Compute librosa melspectrogram. See librosa for params.
    """
    y, S, _ = _frames_helper(y, y_frames, n_fft, 2.0)

    y_melspec = melspectrogram(y=y,
                               S=S,
                               sr=sr,
                               n_fft=n_fft,
                               hop_length=hop_size,
                               fmin=fmin,
                               n_mels=n_mels,
                               center=False)
    return y_melspec.astype(np.float32)


def _logspec_matrix(bins_per_octave, num_bins, f_min, fft_len, sr):
    """
    Compute mixing matrix for log filterbank

    Parameters
    ----------
    bins_per_octave : int
    num_bins : int
    f_min : float
    fft_len : int
    sr : float

    Returns
    -------
    c_mat
    """
    # note range goes from -1 to bpo*num_oct for boundary issues
    f_cq = f_min * 2.0 ** ((np.arange(-1, num_bins+1)) / float(bins_per_octave))
    # centers in bins
    kc = np.round(f_cq * (fft_len / float(sr))).astype(int)
    c_mat = np.zeros([num_bins, int(np.round(fft_len / 2.0))])
    for k in range(1, kc.shape[0]-1):
        l1 = kc[k]-kc[k-1]
        w1 = scipy.signal.triang((l1 * 2) + 1)
        l2 = kc[k+1]-kc[k]
        w2 = scipy.signal.triang((l2 * 2) + 1)
        wk = np.hstack([w1[0:l1], w2[l2:]])  # concatenate two halves
        c_mat[k-1, kc[k-1]:(kc[k+1]+1)] = wk / np.sum(wk)  # normalized to unit sum;
    return c_mat


def _onset_patterns_params(sr, f_hop_size, f_win_size, p_hop_size, p_win_size, mean_filter_size):
    """
    Calculate onset patterns parameters
    """
    if f_hop_size is None:
        # if not specified, set to 0.01 seconds, this hop size is twice as frequent as Holzapfel's,
        # but matches Juan's code.
        f_hop_size = int(2 ** np.ceil(np.log2(0.01 * sr)))
    if f_win_size is None:
        # if not specified, set to 0.04 seconds (this was 0.046s at 44100Hz when rounded, as in Holzapfel's)
        f_win_size = f_hop_size * 4
    p_sr = sr / float(f_hop_size)
    if p_hop_size is None:
        # if not specified, set to 0.5 seconds
        p_hop_size = int(np.round(0.5 * p_sr))
    if p_win_size is None:
        # if not specified, set to 8 seconds. Holzapfel said that more than 8 seconds was detrimental because we want
        # stationarity
        p_win_size = int(np.round(8 * p_sr))
    if mean_filter_size is None:
        # if not specified, set to 0.25 seconds
        mean_filter_size = int(round(0.25 * p_sr))

    return f_hop_size, f_win_size, p_sr, p_hop_size, p_win_size, mean_filter_size


def _onset_detection_fn(x, f_win_size, f_hop_size, f_bins_per_octave, f_octaves, f_fmin, sr, mean_filter_size):
    """
    Filter bank for onset pattern calculation
    """
    # calculate frequency constant-q transform
    f_win = scipy.signal.hanning(f_win_size)
    x_spec = stft(x,
                  n_fft=f_win_size,
                  hop_length=f_hop_size,
                  win_length=f_win_size,
                  window=f_win)
    x_spec = np.abs(x_spec) / float(2 * np.sum(f_win))

    f_cq_mat = _logspec_matrix(f_bins_per_octave, f_octaves * f_bins_per_octave, f_fmin, f_win_size, sr)
    x_cq_spec = np.dot(f_cq_mat, x_spec[:-1, :])

    # subtract moving mean
    b = np.concatenate([[1], np.ones(mean_filter_size, dtype=float) / -mean_filter_size])
    od_fun = scipy.signal.lfilter(b, 1, x_cq_spec, axis=1)

    # half-wave rectify
    od_fun = np.maximum(0, od_fun)

    # post-process OPs
    od_fun = np.log10(1 + 1000*od_fun)
    return od_fun, x_cq_spec


def onset_patterns(x,
                   sr,
                   f_hop_size=None,
                   f_win_size=None,
                   f_fmin=150,
                   f_octaves=6,
                   f_bins_per_octave=10,
                   mean_filter_size=None,
                   p_hop_size=None,
                   p_win_size=None,
                   p_fmin=0.5,
                   p_octaves=5,
                   p_bins_per_octave=12,
                   aggregate_fn=np.mean):
    """
    Calculate the onset patterns rhythm feature. In Holzapfel[1] and Pohle[2], they use a sampling rate of 22050 Hz.
    f_ parameters are for the first logf transform (time domain to frequency domain). p_ parameters are for the second
    logf transform (performed on each band of onset signals to represent onset periodicity).

    References
    ----------


    Parameters
    ----------
    x
    sr
    f_hop_size
    f_win_size
    f_fmin
    f_octaves
    f_bins_per_octave
    mean_filter_size
    p_hop_size
    p_win_size
    p_fmin
    p_octaves
    p_bins_per_octave
    aggregate_fn

    Returns
    -------
    ops
    """
    f_hop_size, f_win_size, p_sr, p_hop_size, p_win_size, mean_filter_size = _onset_patterns_params(sr,
                                                                                                    f_hop_size,
                                                                                                    f_win_size,
                                                                                                    p_hop_size,
                                                                                                    p_win_size,
                                                                                                    mean_filter_size)

    # normalize
    x /= np.max(np.abs(x))

    od_fun, x_cq_spec = _onset_detection_fn(x, f_win_size, f_hop_size, f_bins_per_octave, f_octaves, f_fmin, sr, mean_filter_size)

    # calculate periodicity constant-q transform
    ops = np.empty([od_fun.shape[0], p_octaves * p_bins_per_octave, int(np.ceil(od_fun.shape[1] / float(p_hop_size)))])
    p_cq_mat = _logspec_matrix(p_bins_per_octave, p_octaves * p_bins_per_octave, p_fmin, p_win_size, p_sr)
    p_win = scipy.signal.hanning(p_win_size)
    for i in range(od_fun.shape[0]):
        od_spec = stft(od_fun[i, :],
                       n_fft=p_win_size,
                       hop_length=p_hop_size,
                       win_length=p_win_size,
                       window=p_win)
        od_spec = np.abs(od_spec) / (2 * np.sum(p_win))
        od_cq_spec = np.dot(p_cq_mat, od_spec[:-1, :])
        ops[i, :, :] = od_cq_spec[:ops[i, :, :].shape[0], :ops[i, :, :].shape[1]]

    # aggregate ops if `aggregate_fn` is not None
    if aggregate_fn is not None:
        ops = aggregate_fn(ops, axis=2)

    return ops


def linspec(y=None, y_frames=None, n_fft=2048, hop_size=512, return_angle=True):
    """
    Magnitude of linear spectrum
    """
    win = scipy.signal.hanning(n_fft)

    if y_frames is not None:
        S = frames_stft(y_frames=y_frames,
                        n_fft=n_fft,
                        window=win)
    else:
        if y is None:
            raise Exception('y or frames must be defined.')
        S = stft(y,
                 n_fft=n_fft,
                 hop_length=hop_size,
                 window=win,
                 center=False)

    Sm = np.abs(S).astype(np.float32)

    if return_angle:
        Sp = np.angle(S).astype(np.float32)
    else:
        Sp = None

    return Sm, Sp


def logspec(y=None, y_frames=None, sr=22050, n_fft=1024, hop_size=221, f_min=40.0, bins_per_octave=8, n_octaves=8,
            log_mat=None):
    """
    Magnitude of logf-spectrogram
    """
    win = scipy.signal.hanning(n_fft)

    if y_frames is not None:
        S = frames_stft(y_frames=y_frames,
                        n_fft=n_fft,
                        window=win,
                        center=False)
    else:
        if y is None:
            raise Exception('y or frames must be defined.')

        win = scipy.signal.hanning(n_fft)
        S = stft(y,
                 n_fft=n_fft,
                 hop_length=hop_size,
                 window=win,
                 center=False)

    y_spec = np.abs(S) / (2 * np.sum(win))

    if log_mat is None:
        log_mat = _logspec_matrix(bins_per_octave, n_octaves * bins_per_octave, f_min, n_fft, sr)
    y_logspec = np.dot(log_mat, y_spec[:-1, :])

    return y_logspec.astype(np.float32)


def hcqt(y, sr=22050, hop_size=256, fmin=32.7, bins_per_octave=60, n_octaves=6, harmonics=(0.5, 1, 2, 3, 4, 5)):
    """
    Harmonic CQT. Compute CQT at harmonics of `fmin`. See librosa for cqt params.
    """

    cqt_mag, cqt_phase = [], []

    n_frames = time_to_frames(get_duration(y=y, sr=sr), sr=sr, hop_length=hop_size)

    for h in harmonics:
        y_cqt = cqt(y=y,
                    sr=sr,
                    hop_length=hop_size,
                    fmin=fmin * h,
                    n_bins=n_octaves * bins_per_octave,
                    bins_per_octave=bins_per_octave,
                    res_type='kaiser_best')

        y_cqt = fix_length(y_cqt, n_frames)

        y_cqt_mag, y_cqt_phase = magphase(y_cqt)

        cqt_mag.append(y_cqt_mag)
        cqt_phase.append(y_cqt_phase)

    cqt_mag = np.asarray(cqt_mag).astype(np.float32)
    cqt_phase = np.angle(np.asarray(cqt_phase)).astype(np.float32)

    return cqt_mag, cqt_phase


def vggish_melspec(y, sr=22050, do_resample=False, frames=None):
    """
    Extract melspec for vggish model
    """
    if sr != vggish_params.SAMPLE_RATE and do_resample:
        if frames is not None:
            raise Exception("Resampled not supported with frames argument.")
        y = resample(y, sr, vggish_params.SAMPLE_RATE)
        sr = vggish_params.SAMPLE_RATE

    log_mel = mel_features.log_mel_spectrogram(y,
                                               frames=frames,
                                               audio_sample_rate=sr,
                                               log_offset=vggish_params.LOG_OFFSET,
                                               window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
                                               hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
                                               num_mel_bins=vggish_params.NUM_MEL_BINS,
                                               lower_edge_hertz=vggish_params.MEL_MIN_HZ,
                                               upper_edge_hertz=vggish_params.MEL_MAX_HZ)
    return log_mel


def extract(args):
    audio_directory, output_directory, af, overwrite = args
    subdir, output_file = os.path.split(af.split(audio_directory)[1])
    output_file = os.path.splitext(output_file)[0]
    output_file = os.path.join(output_directory, output_file)

    if os.path.exists(output_file) and not overwrite:
        print('Skipping {}. Already exists.'.format(output_file))
        return

    output = dict()

    try:
        y, _sr = soundfile.read(af)
        y = to_mono(y)
        sr = 22050
        y = resample(y, _sr, sr)
    except Exception as e:
        y, sr = load(af)

    output['linspec_mag'], output['linspec_phase'] = linspec(y)
    output['melspec'] = melspec(y, sr=sr)
    output['logspec'] = logspec(y, sr=sr)
    output['hcqt_mag'], output['hcqt_phase'] = hcqt(y, sr=sr)
    output['vggish_melspec'] = vggish_melspec(y, sr=sr)

    # high-level
    output['percussive_ratio'], output['percussive_rms'], output['total_rms'] = percussive_ratio(y, margin=3.0)
    output['onset_strength'] = onset_strength(y, detrend=True)
    output['tempogram'] = tempogram(y)
    output['onset_patterns'] = onset_patterns(y, sr=sr)

    np.savez_compressed(output_file, **output)


if __name__ == '__main__':
    import soundfile
    import tqdm

    params = process_arguments(sys.argv[1:])

    audio_files = []
    for root, dirnames, filenames in os.walk(params.audio_directory):
        for filename in filenames:
            if filename.endswith(('.wav', '.mp3', '.flv')):
                audio_files.append(os.path.join(root, filename))

    if not os.path.exists(params.output_directory):
        os.makedirs(params.output_directory)

    p = Pool(processes=params.cpus)
    max_ = len(audio_files)
    ret = p.map(extract, [(params.audio_directory, params.output_directory, af, params.overwrite) for af in audio_files])



