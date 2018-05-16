import numpy as np
import random


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


class PickClosestLengthStrategy(Strategy):
    def train(self, data: np.ndarray) -> None:
        pass

    def predict(self, data: np.ndarray) -> str:
        first_option = data[5]
        second_option = data[6]
        sentence_lengths = [len(x) for x in data[1:5]]
        average_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
        if abs(average_sentence_length - len(first_option)) <= abs(average_sentence_length - len(second_option)):
            return '1'
        else:
            return '2'


class ConstantPickLongestStrategy(Strategy):
    def train(self, data: np.ndarray) -> None:
        pass

    def predict(self, data: np.ndarray) -> str:
        first_option = data[5]
        second_option = data[6]
        if len(first_option) >= len(second_option):
            return '1'
        else:
            return '2'


class PickRandomStrategy(Strategy):
    def train(self, data: np.ndarray) -> None:
        pass

    def predict(self, data: np.ndarray) -> str:
        return str(random.randint(1, 2))
