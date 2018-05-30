import numpy as np
from strategies import Strategy


class Evaluator(object):
    @staticmethod
    def validation_error(strategy: Strategy, training_data: np.ndarray, validation_data: np.ndarray) -> float:
        val_stories, val_labels = np.split(validation_data, [-1], axis=1 )
        val_labels = np.squeeze(val_labels)

        strategy.fit(training_data)
        predictions = strategy.predict(val_stories)

        return np.mean(np.equal(predictions, val_labels.astype(int)))