from abc import ABCMeta, abstractmethod
from peach.utils.hparams import HParams


class ModelStructure(metaclass=ABCMeta):
    @staticmethod
    def get_default_model_parameters():
        return HParams()

    @staticmethod
    def get_identity_param_list():
        return []

