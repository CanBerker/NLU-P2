import readers
import os
import numpy as np

from utils.loader import join_and_shuffle
from utils.loader import get_labels
from evaluators import Evaluator
from evaluators.per_dp_evaluator import PerDataPointEvaluator
from evaluators.topic_discovery_evaluator import TopicDiscoveryEvaluator
from evaluators.sentiment_trajectory_evaluator import OnlyValidationDataEvaluator

from strategies.sentiment_trajectory import SentimentTrajectoryStrategy
from strategies.stylistic_features import StylisticFeaturesStrategy
from strategies.topic_discovery import TopicDiscoveryStrategy
from strategies.sklearn_nb import NBStrategy
from strategies.language_model import LanguageModelStrategy
from strategies.topic_consistency import TopicConsistencyStrategy
from strategies.lstm_classifier import LSTMClassifierStrategy
from data_augmentation import augment_data


if __name__ == '__main__':
    train_data_loc = os.path.join(os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'), 'train_small.csv')
    validation_data_loc = os.path.join(os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'), 'validation.csv')

    print('Train data location: {}'.format(train_data_loc))
    print('Validation data location: {}'.format(validation_data_loc))

    train_data = readers.TrainReader(train_data_loc).read()
    validation_data = readers.ValidationReader(validation_data_loc).read()

    negative_samples = augment_data(train_data, 1, load=False)
    positive_samples = train_data
    
    negative_lables = get_labels(negative_samples, 0)
    positive_labels = get_labels(positive_samples, 1)
    
    all_stories, all_labels = join_and_shuffle([negative_samples, positive_samples],
                                            [negative_lables, positive_labels])
    
    all_data = np.column_stack((all_stories, all_labels))
    
    print(np.array(all_stories).shape, np.array(all_labels).shape, all_data.shape)

    #strategy = TopicDiscoveryStrategy(TopicDiscoveryEvaluator())
    #strategy = SentimentTrajectoryStrategy(SentimentTrajectoryEvaluator())
    #strategy = NBStrategy(PerDataPointEvaluator())
    #strategy = StylisticFeaturesStrategy(OnlyValidationDataEvaluator())
    # strategy = LanguageModelStrategy(Evaluator())
    strategy = LSTMClassifierStrategy(Evaluator())
    validation_error = strategy.evaluator.validation_error(strategy, all_data, validation_data)
    #validation_error = Evaluator.validation_error(strategy, train_data, validation_data)
    print('Validation error: {}'.format(validation_error))
