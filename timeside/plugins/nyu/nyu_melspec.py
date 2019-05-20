# -*- coding: utf-8 -*-

from timeside.core import implements, interfacedoc
from timeside.core.preprocessors import downmix_to_mono, frames_adapter
from timeside.core.tools.parameters import store_parameters
from timeside.core.analyzer import Analyzer
from timeside.core.api import IAnalyzer
import numpy as np

from features import melspec


class NYUMelSpectrogam(Analyzer):

    """Mel spectrogram"""
    implements(IAnalyzer)

    @store_parameters
    def __init__(self,
                 input_blocksize=2048,
                 input_stepsize=512,
                 input_samplerate=22050,
                 fft_size=2048,
                 n_mels=128,
                 fmin=0.0):
        super(NYUMelSpectrogam, self).__init__()
        self.input_blocksize = input_blocksize
        self.input_stepsize = input_stepsize
        self.input_samplerate = input_samplerate
        self.fft_size = fft_size
        self.n_mels = n_mels
        self.fmin = fmin


    @interfacedoc
    def setup(self, channels=None,
              samplerate=None,
              blocksize=None,
              totalframes=None):
        super(NYUMelSpectrogam, self).setup(channels, samplerate, blocksize, totalframes)
        self.values = []


    @staticmethod
    @interfacedoc
    def id():
        return "nyu_melspec"

    @staticmethod
    @interfacedoc
    def name():
        return "NYU Mel Spectrogram"

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
        self.values.append(frames)
        return frames, eod

    def post_process(self):
        self.result = self.new_result(data_mode='value', time_mode='framewise')

        self.result.data_object.value = melspec(y_frames=np.vstack(self.values).T,
                            sr=self.samplerate(),
                            n_fft=self.input_blocksize,
                            hop_size=self.input_stepsize,
                            n_mels=self.n_mels,
                            fmin=self.fmin)

        self.add_result(self.result)
