import numpy as np


class Strategy(object):

    def __init__(self, evaluator, save_path, use_gpu, glove_path=None):
        self.use_gpu = use_gpu
        self.evaluator = evaluator
        self.glove_path = glove_path
        self.save_path = save_path

    def fit(self, data: np.ndarray) -> None:
        raise NotImplementedError("Strategy not implemented.")

    def predict(self, data: np.ndarray) -> str:
        raise NotImplementedError("Strategy not implemented.")

    def log(self, line):
        print("[{0}] {1}".format(self.__class__.__name__, line))
