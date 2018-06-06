import numpy as np
from strategies import Strategy


class TopicDiscoveryEvaluator(object):
    def __init__(self, training_data: np.ndarray, validation_data: np.ndarray):
        self.training_data = training_data
        self.validation_data = validation_data

    def validation_error(self, strategy: Strategy) -> float:
        val_stories, val_labels = np.split(self.validation_data, [-1], axis=1 )
        val_labels = np.squeeze(val_labels)

        strategy.fit(self.training_data, self.validation_data)
        predictions = strategy.predict(val_stories)

        return np.mean(np.equal(predictions, val_labels.astype(int)))
