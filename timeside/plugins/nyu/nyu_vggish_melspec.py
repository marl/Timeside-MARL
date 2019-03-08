# -*- coding: utf-8 -*-

from timeside.core import implements, interfacedoc
from timeside.core.preprocessors import downmix_to_mono, frames_adapter
from timeside.core.tools.parameters import store_parameters
from timeside.core.analyzer import Analyzer
from timeside.core.api import IAnalyzer
import numpy as np

from features import vggish_melspec


class NYUVGGishMelSpectrogam(Analyzer):

    """Mel spectrogram"""
    implements(IAnalyzer)

    @store_parameters
    def __init__(self,
                 input_blocksize=256,
                 input_stepsize=256,
                 input_samplerate=16000):
        super(NYUVGGishMelSpectrogam, self).__init__()
        self.input_blocksize = input_blocksize
        self.input_stepsize = input_stepsize
        self.input_samplerate = input_samplerate


    @interfacedoc
    def setup(self, channels=None,
              samplerate=None,
              blocksize=None,
              totalframes=None):
        super(NYUVGGishMelSpectrogam, self).setup(channels, samplerate, blocksize, totalframes)
        self.values = []


    @staticmethod
    @interfacedoc
    def id():
        return "nyu_vggish_melspec"

    @staticmethod
    @interfacedoc
    def name():
        return "NYU VGGish Mel Spectrogram"

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
        result = self.new_result(data_mode='value', time_mode='framewise')

        y = np.hstack(self.values)

        y_vggish_melspec = vggish_melspec(y=y,
                                          sr=self.samplerate(), )

        result.data_object.value = y_vggish_melspec
        self.add_result(result)
