# -*- coding: utf-8 -*-

from timeside.core import implements, interfacedoc
from timeside.core.preprocessors import downmix_to_mono, frames_adapter
from timeside.core.tools.parameters import store_parameters
from timeside.core.analyzer import Analyzer
from timeside.core.api import IAnalyzer
import numpy as np

from features import linspec


class NYULinearSpectrogam(Analyzer):

    """Linear spectogram"""
    implements(IAnalyzer)

    @store_parameters
    def __init__(self,
                 input_blocksize=2048,
                 input_stepsize=512,
                 input_samplerate=22050,
                 fft_size=2048):
        super(NYULinearSpectrogam, self).__init__()
        self.input_blocksize = input_blocksize
        self.input_stepsize = input_stepsize
        self.input_samplerate = input_samplerate
        self.fft_size = fft_size
        self.frame_idx = 0
        self.values = None


    @interfacedoc
    def setup(self, channels=None,
              samplerate=None,
              blocksize=None,
              totalframes=None):
        super(NYULinearSpectrogam, self).setup(channels, samplerate, blocksize, totalframes)
        totalblocks = (self.totalframes() - self.input_blocksize) / self.input_stepsize + 2
        self.values = np.empty([totalblocks, self.fft_size / 2 + 1])


    @staticmethod
    @interfacedoc
    def id():
        return "nyu_linspec"

    @staticmethod
    @interfacedoc
    def name():
        return "NYU Linear Spectrogram"

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
        y_linspec, _ = linspec(y=frames,
                               n_fft=self.input_blocksize,
                               hop_size=self.input_stepsize,
                               return_angle=False)
        assert(y_linspec.shape[1] == 1)
        self.values[self.frame_idx, :] = y_linspec.reshape(-1)
        self.frame_idx += 1
        return frames, eod

    def post_process(self):
        result = self.new_result(data_mode='value', time_mode='framewise')
        result.data_object.value = self.values
        self.add_result(result)
