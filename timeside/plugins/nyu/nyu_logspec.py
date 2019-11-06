# -*- coding: utf-8 -*-

from timeside.core import implements, interfacedoc
from timeside.core.preprocessors import downmix_to_mono, frames_adapter
from timeside.core.tools.parameters import store_parameters
from timeside.core.analyzer import Analyzer
from timeside.core.api import IAnalyzer
import numpy as np

from features import logspec, _logspec_matrix


class NYULogSpectrogam(Analyzer):

    """Log-f spectrogram"""
    implements(IAnalyzer)

    @store_parameters
    def __init__(self,
                 input_blocksize=1024,
                 input_stepsize=221,
                 input_samplerate=22050,
                 fft_size=1024,
                 f_min=40.0,
                 bins_per_octave=8,
                 n_octaves=8):
        super(NYULogSpectrogam, self).__init__()
        self.input_blocksize = input_blocksize
        self.input_stepsize = input_stepsize
        self.input_samplerate = input_samplerate
        self.fft_size = fft_size
        self.bins_per_octave = bins_per_octave
        self.n_octaves = n_octaves
        self.f_min = f_min
        self.frame_idx = 0
        self.values = None


    @interfacedoc
    def setup(self, channels=None,
              samplerate=None,
              blocksize=None,
              totalframes=None):
        super(NYULogSpectrogam, self).setup(channels, samplerate, blocksize, totalframes)
        totalblocks = (self.totalframes() - self.input_blocksize) / self.input_stepsize + 2
        self.values = np.empty([totalblocks, self.bins_per_octave * self.n_octaves])
        self.log_mat = _logspec_matrix(self.bins_per_octave,
                                       self.n_octaves * self.bins_per_octave,
                                       self.f_min,
                                       self.fft_size, self.input_samplerate)


    @staticmethod
    @interfacedoc
    def id():
        return "nyu_logspec"

    @staticmethod
    @interfacedoc
    def name():
        return "NYU Log Spectrogram"

    @staticmethod
    @interfacedoc
    def unit():
        return ""

    @staticmethod
    @interfacedoc
    def version():
        return '1.0'

    @property
    def force_samplerate(self):
        return self.input_samplerate

    @downmix_to_mono
    @frames_adapter
    def process(self, frames, eod=False):
        y_logspec = logspec(y=frames,
                            n_fft=self.input_blocksize,
                            hop_size=self.input_stepsize,
                            bins_per_octave=self.bins_per_octave,
                            f_min=self.f_min,
                            n_octaves=self.n_octaves,
                            log_mat=self.log_mat)
        assert (y_logspec.shape[1] == 1)
        self.values[self.frame_idx, :] = y_logspec.reshape(-1)
        self.frame_idx += 1
        return frames, eod

    def post_process(self):
        result = self.new_result(data_mode='value', time_mode='framewise')
        result.data_object.value = self.values
        self.add_result(result)
