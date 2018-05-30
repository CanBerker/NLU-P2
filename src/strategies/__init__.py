import numpy as np

class Strategy(object):
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def fit(self, data: np.ndarray) -> None:
        raise NotImplementedError("Strategy not implemented.")

    def predict(self, data: np.ndarray) -> str:
        raise NotImplementedError("Strategy not implemented.")
