import numpy as np
from strategies import Strategy


class PerDataPointEvaluator(object):
    def __init__(self, training_data: np.ndarray, validation_data: np.ndarray):
        self.training_data = training_data
        self.validation_data = validation_data

    def validation_error(self, strategy: Strategy) -> float:
        total = len(self.validation_data)
        correct = 0
        strategy.fit(self.training_data)
        for data_point in self.validation_data:
            prediction = strategy.predict(data_point)
            if prediction == data_point[-1]:
                correct += 1
        return correct / total