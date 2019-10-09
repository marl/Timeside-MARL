# -*- coding: utf-8 -*-

from timeside.core import implements, interfacedoc
from timeside.core.preprocessors import downmix_to_mono, frames_adapter
from timeside.core.tools.parameters import store_parameters
from timeside.core.analyzer import Analyzer
from timeside.core.api import IAnalyzer
import numpy as np

from features import hcqt
import librosa.filters as filters


class NYUHCQT(Analyzer):

    """Harmonic Constant-Q Transform"""
    implements(IAnalyzer)

    @store_parameters
    def __init__(self,
                 input_blocksize=256,
                 input_stepsize=256,
                 input_samplerate=22050,
                 fmin=32.7,
                 bins_per_octave=60,
                 n_octaves=6,
                 harmonics=(0.5, 1, 2, 3, 4, 5),
                 buffer_size=1000):
        super(NYUHCQT, self).__init__()
        self.input_blocksize = input_blocksize
        self.input_stepsize = input_stepsize
        self.input_samplerate = input_samplerate
        self.fmin = fmin
        self.bins_per_octave = bins_per_octave
        self.n_octaves = n_octaves
        self.harmonics = harmonics
        self.buffer_size = buffer_size

        lengths = filters.constant_q_lengths(self.input_samplerate,
                                             fmin=self.fmin,
                                             n_bins=self.n_octaves * self.bins_per_octave,
                                             bins_per_octave=self.bins_per_octave,
                                             tuning=0.0,
                                             window='hann',
                                             filter_scale=1)

        self.buffer_margin_size = int(round(lengths[0] / self.input_blocksize))
        self.buffer = np.zeros(self.input_blocksize * self.buffer_size)
        self.idx = self.buffer_margin_size * self.input_blocksize

        self.cleanup = False

        self.output_idx = 0
        self.values = None


    @interfacedoc
    def setup(self, channels=None,
              samplerate=None,
              blocksize=None,
              totalframes=None):
        super(NYUHCQT, self).setup(channels, samplerate, blocksize, totalframes)
        totalblocks = (self.totalframes() - self.input_blocksize) / self.input_stepsize + 2
        self.values = np.empty([totalblocks, len(self.harmonics), self.bins_per_octave * self.n_octaves])


    @staticmethod
    @interfacedoc
    def id():
        return "nyu_hcqt"

    @staticmethod
    @interfacedoc
    def name():
        return "NYU Harmonic Constant-Q Transform"

    @staticmethod
    @interfacedoc
    def unit():
        return ""

    @property
    def force_samplerate(self):
        return self.input_samplerate

    @downmix_to_mono
    @frames_adapter
    def process(self, frames, eod=False):
        self.buffer[self.idx:(self.idx + frames.shape[0])] = frames
        self.idx += frames.shape[0]

        if self.idx == (self.buffer_size * self.input_blocksize):
            y_hcqt, _ = hcqt(self.buffer, sr=self.samplerate(), hop_size=self.input_stepsize, fmin=self.fmin,
                             bins_per_octave=self.bins_per_octave, n_octaves=self.n_octaves,
                             harmonics=self.harmonics)
            out = y_hcqt[:, :, self.buffer_margin_size:(self.buffer_size - self.buffer_margin_size)]
            out = np.moveaxis(out, -1, 0)
            self.values[self.output_idx:self.output_idx + out.shape[0], :, :] = out
            self.output_idx += out.shape[0]

            offset = 2 * self.buffer_margin_size * self.input_blocksize
            self.buffer = np.roll(self.buffer, offset)
            self.idx = offset
            self.cleanup = False
        else:
            self.cleanup = True

        return frames, eod

    def post_process(self):
        result = self.new_result(data_mode='value', time_mode='framewise')

        if self.cleanup:
            y_hcqt, _ = hcqt(self.buffer[:self.idx], sr=self.samplerate(), hop_size=self.input_stepsize, fmin=self.fmin,
                             bins_per_octave=self.bins_per_octave, n_octaves=self.n_octaves,
                             harmonics=self.harmonics)
            out = y_hcqt[:, :, self.buffer_margin_size:(self.buffer_size - self.buffer_margin_size)]
            out = np.moveaxis(out, -1, 0)
            self.values[self.output_idx:self.output_idx + out.shape[0], :, :] = out

        self.values = np.dstack(self.values)
        self.values = np.moveaxis(self.values, -1, 0)
        result.data_object.value = self.values
        self.add_result(result)
