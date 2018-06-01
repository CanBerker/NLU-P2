import readers
import os

from evaluators import Evaluator
from evaluators.per_dp_evaluator import PerDataPointEvaluator
from evaluators.topic_discovery_evaluator import TopicDiscoveryEvaluator
from evaluators.sentiment_trajectory_evaluator import OnlyValidationDataEvaluator

from strategies.sentiment_trajectory import SentimentTrajectoryStrategy
from strategies.stylistic_features import StylisticFeaturesStrategy
from strategies.topic_discovery import TopicDiscoveryStrategy
from strategies.sklearn_nb import NBStrategy
from strategies.language_model import LanguageModelStrategy
from data_augmentation import augment_data


if __name__ == '__main__':
    train_data_loc = os.path.join(os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'), 'train.csv')
    validation_data_loc = os.path.join(os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'), 'validation.csv')

    print('Train data location: {}'.format(train_data_loc))
    print('Validation data location: {}'.format(validation_data_loc))

    train_data = readers.TrainReader(train_data_loc).read()
    validation_data = readers.ValidationReader(validation_data_loc).read()

    augmented_data = augment_data(train_data, 4)

    #strategy = TopicDiscoveryStrategy(TopicDiscoveryEvaluator())
    #strategy = SentimentTrajectoryStrategy(SentimentTrajectoryEvaluator())
    #strategy = NBStrategy(PerDataPointEvaluator())
    #strategy = StylisticFeaturesStrategy(OnlyValidationDataEvaluator())
    strategy = LanguageModelStrategy(Evaluator())
    validation_error = strategy.evaluator.validation_error(strategy, train_data, validation_data)
    #validation_error = Evaluator.validation_error(strategy, train_data, validation_data)
    print('Validation error: {}'.format(validation_error))
