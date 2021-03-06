import numpy as np
from strategies import Strategy


class EnsembleEvaluator(object):
    def __init__(self, training_data: np.ndarray,
                 validation_data: np.ndarray, augmented_data: np.ndarray, validate_on_eth = False):
        self.training_data = training_data
        self.validation_data = validation_data
        self.augmented_data = augmented_data
        self.validate_on_eth = validate_on_eth

    def validation_error(self, strategy: Strategy) -> float:
        # Take off labels
        val_stories, val_labels = np.split(self.validation_data, [-1], axis=1 )
        val_labels = np.squeeze(val_labels)

        # Strategy can fit on augmented or usual training data.
        strategy.fit(self.training_data, self.augmented_data)
        predictions = strategy.predict(val_stories)

        #When predicting the ETH custom set (need to write out)
        if self.validate_on_eth:
            with open("nlu_test.out", "w+") as f:
                [f.write("{0}\n".format(p)) for p in predictions]

        pred = 0.0
        if not self.validate_on_eth:
            pred = np.mean(np.equal(predictions, val_labels.astype(int)))
        
        return pred
