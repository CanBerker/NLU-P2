import numpy as np
from strategies import Strategy


class PerDataPointEvaluator(object):
    @staticmethod
    def validation_error(strategy: Strategy, training_data: np.ndarray, validation_data: np.ndarray) -> float:
        total = len(validation_data)
        correct = 0
        strategy.fit(training_data)
        for data_point in validation_data:
            prediction = strategy.predict(data_point)
            if prediction == data_point[-1]:
                correct += 1
        return correct / total