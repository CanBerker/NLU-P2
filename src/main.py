import readers
from evaluator import Evaluator
from strategies.lstm import CharacterBasedLstmStrategy

from strategies.sentiment_trajectory import SentimentTracjectoryStrategy
from strategies.stylistic_features   import StylisticFeaturesStrategy
from strategies.sklearn_nb           import NBStrategy
from strategies.topic_discovery      import TopicDiscoveryStrategy

import os

if __name__ == '__main__':
    train_data_loc = os.path.join(os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'), 'train.csv')
    validation_data_loc = os.path.join(os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'), 'validation.csv')

    print('Train data location: {}'.format(train_data_loc))
    print('Validation data location: {}'.format(validation_data_loc))

    train_data = readers.TrainReader(train_data_loc).read()
    validation_data = readers.ValidationReader(validation_data_loc).read()
    strategy = TopicDiscoveryStrategy()
    validation_error = Evaluator.validation_error(strategy, train_data, validation_data)
    print('Validation error: {}'.format(validation_error))
