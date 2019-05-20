# -*- coding: utf-8 -*-

from timeside.core import implements, interfacedoc
from timeside.core.preprocessors import downmix_to_mono, frames_adapter
from timeside.core.tools.parameters import store_parameters
from timeside.core.analyzer import Analyzer
from timeside.core.api import IAnalyzer
import numpy as np

from features import hcqt


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
                 harmonics=(0.5, 1, 2, 3, 4, 5)):
        super(NYUHCQT, self).__init__()
        self.input_blocksize = input_blocksize
        self.input_stepsize = input_stepsize
        self.input_samplerate = input_samplerate
        self.fmin = fmin
        self.bins_per_octave = bins_per_octave
        self.n_octaves = n_octaves
        self.harmonics = harmonics


    @interfacedoc
    def setup(self, channels=None,
              samplerate=None,
              blocksize=None,
              totalframes=None):
        super(NYUHCQT, self).setup(channels, samplerate, blocksize, totalframes)
        self.values = []


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
        self.values.append(frames)
        return frames, eod

    def post_process(self):
        self.result = self.new_result(data_mode='value', time_mode='framewise')

        self.result.data_object.value, _ = hcqt(np.hstack(self.values), 
                         sr=self.samplerate(), hop_size=self.input_stepsize, fmin=self.fmin,
                         bins_per_octave=self.bins_per_octave, n_octaves=self.n_octaves,
                         harmonics=self.harmonics)

        self.add_result(self.result)
