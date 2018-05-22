import numpy as np
import random
import nltk.sentiment as sent

from strategies import Strategy

class SentimentTracjectoryStrategy(Strategy):
    def fit(self, data: np.ndarray) -> None:
        print(data[0])
        trajectories = self.find_trajectories(data)
        pass

    def predict(self, data: np.ndarray) -> str:
        return '1'
        
    def find_trajectories(self, stories):
        return None