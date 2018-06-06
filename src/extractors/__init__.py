import numpy as np


class Extractor(object):

    def fit(self, data: np.ndarray) -> None:
        raise NotImplementedError("Strategy not implemented.")

    def extract(self, data: np.ndarray) -> str:
        raise NotImplementedError("Strategy not implemented.")

    def log(self, line):
        print("[{0}] {1}".format(self.__class__.__name__, line))