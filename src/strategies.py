import numpy as np


class Strategy(object):
    def train(self, data: np.ndarray) -> None:
        raise NotImplementedError("Strategy not implemented.")

    def predict(self, data: np.ndarray) -> str:
        raise NotImplementedError("Strategy not implemented.")


class ConstantChooseFirstStrategy(Strategy):
    def train(self, data: np.ndarray) -> None:
        pass

    def predict(self, data: np.ndarray) -> str:
        return '1'
