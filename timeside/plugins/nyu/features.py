import argparse
import os
import sys

import numpy as np
import scipy.signal
import scipy.fftpack as fft

from librosa import cqt, magphase, note_to_hz, stft, resample, to_mono
from librosa import amplitude_to_db, get_duration, time_to_frames
from librosa.util import fix_length
from librosa.feature import melspectrogram

from librosa.filters import get_window
from librosa.core.audio import resample
from librosa import util


def process_arguments(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('audio_directory', type=str,
                        help="The path to the directory of audio files.")

    parser.add_argument('output_directory', type=str,
                        help="Path to output directory.")

    return parser.parse_args(args)


def _frames_helper(y, y_frames, n_fft, power=1.0):
    if y_frames is not None:
        S = frames_stft(y_frames=y_frames,
                        n_fft=n_fft)

        Sm = np.abs(S) ** power
        Sp = np.angle(S)
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
    n_columns = int(util.MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]]

    return stft_matrix


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
                               n_mels=n_mels)
    return y_melspec.astype(np.float32)


def _logspec_matrix(bins_per_octave, num_bins, f_min, fft_len, sr):
    """
    Compute center frequencies of the log-f filterbank

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
    kc = np.round(f_cq * (fft_len / sr)).astype(int)
    c_mat = np.zeros([num_bins, int(np.round(fft_len / 2))])
    for k in range(1, kc.shape[0]-1):
        l1 = kc[k]-kc[k-1]
        w1 = scipy.signal.triang((l1 * 2) + 1)
        l2 = kc[k+1]-kc[k]
        w2 = scipy.signal.triang((l2 * 2) + 1)
        wk = np.hstack([w1[0:l1], w2[l2:]])  # concatenate two halves
        c_mat[k-1, kc[k-1]:(kc[k+1]+1)] = wk / np.sum(wk)  # normalized to unit sum;
    return c_mat


def linspec(y=None, y_frames=None, n_fft=2048, hop_size=512):
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
                 window=win)

    Sm = np.abs(S)
    Sp = np.angle(S)

    return Sm.astype(np.float32), Sp.astype(np.float32)


def logspec(y=None, y_frames=None, sr=22050, n_fft=1024, hop_size=221, fmin=0.0, bins_per_octave=8, n_octaves=8):
    """
    Magnitude of logf-spectrogram
    """
    win = scipy.signal.hanning(n_fft)

    if y_frames is not None:
        S = frames_stft(y_frames=y_frames,
                        n_fft=n_fft,
                        window=win)
    else:
        if y is None:
            raise Exception('y or frames must be defined.')

        win = scipy.signal.hanning(n_fft)
        S = stft(y,
                 n_fft=n_fft,
                 hop_length=hop_size,
                 window=win)

    y_spec = np.abs(S) / (2 * np.sum(win))

    log_mat = _logspec_matrix(bins_per_octave, n_octaves * bins_per_octave, fmin, n_fft, sr)
    y_logspec = np.dot(log_mat, y_spec[:-1, :])

    return y_logspec.astype(np.float32)


def hcqt(y, sr=22040, hop_size=256, fmin=32.7, bins_per_octave=60, n_octaves=6, harmonics=(0.5, 1, 2, 3, 4, 5)):
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
                    bins_per_octave=bins_per_octave)

        y_cqt = fix_length(y_cqt, n_frames)

        y_cqt_mag, y_cqt_phase = magphase(y_cqt)

        cqt_mag.append(y_cqt_mag)
        cqt_phase.append(y_cqt_phase)

    cqt_mag = np.asarray(cqt_mag).astype(np.float32)
    cqt_phase = np.angle(np.asarray(cqt_phase)).astype(np.float32)

    return cqt_mag, cqt_phase


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

    for af in tqdm.tqdm(audio_files):
        output = dict()

        y, _sr = soundfile.read(af)
        y = to_mono(y)
        sr = 22050
        y = resample(y, _sr, sr)

        output['y_linspec_mag'], output['y_linspec_phase'] = linspec(y)
        output['y_melspec'] = melspec(y, sr=sr)
        output['y_logspec'] = logspec(y, sr=sr)
        output['y_hcqt_mag'], output['y_hcqt_phase'] = hcqt(y, sr=sr)

        subdir, output_file = os.path.split(af.split(params.audio_directory)[1])
        output_file = os.path.splitext(output_file)[0]
        output_file = os.path.join(params.output_directory, output_file)

        np.savez_compressed(output_file, **output)


