import numpy as np
import readers
import sys
import os

from argparse import ArgumentParser

from pathlib import Path

from utils.loader import join_and_shuffle
from utils.loader import get_labels
from evaluators import Evaluator
from evaluators.per_dp_evaluator import PerDataPointEvaluator
from evaluators.topic_discovery_evaluator import TopicDiscoveryEvaluator
from evaluators.sentiment_trajectory_evaluator import OnlyValidationDataEvaluator
from evaluators.ensemble_evaluator import EnsembleEvaluator

from strategies.sentiment_trajectory import SentimentTrajectoryStrategy
from strategies.stylistic_features import StylisticFeaturesStrategy
from strategies.topic_discovery import TopicDiscoveryStrategy
from strategies.sklearn_nb import NBStrategy
from strategies.language_model import LanguageModelStrategy
from strategies.topic_consistency import TopicConsistencyStrategy
from strategies.lstm_classifier import LSTMClassifierStrategy
from strategies.ensemble import EnsembleStrategy

from data_augmentation import augment_data


if __name__ == '__main__':
    np.random.seed(135511)
    
    train_data_loc = os.path.join(os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'), 'train_small.csv')
    validation_data_loc = os.path.join(os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'), 'validation.csv')
    glove_file = "glove.6B.50d.txt"
    save_path = "checkpoint"

    parser = ArgumentParser()
    parser.add_argument("-t", dest='tpath', help="Training dataset path")
    parser.add_argument("-v", dest='vpath', help="Validation dataset path")
    parser.add_argument("-s", dest='spath', default=save_path, help="Model directory")
    parser.add_argument("-m", dest='model', help="Model to be trained")
    parser.add_argument("-gp", dest='glove_path', help="Glove file path")
    parser.add_argument("-g", dest='use_gpu', action='store_true', help="Use GPU for training")
    parser.add_argument("-en", dest='use_ensemble', action='store_true', help="Use ensemble of methods")
    parser.add_argument("-ct", dest='continue_training', action='store_true', default=False, help="Continue training")
    parser.add_argument("-mp", dest='model_path', default=None, help="Model path to continue training")

    args = parser.parse_args()
    print(args)
    print(args.glove_path)
    if not Path(args.spath).is_dir():
        print("THERE IS NO SAVE DIRECTORY!!!")
        print("{0} does NOT EXISTS!".format(args.spath))
        sys.exit(1)

    if args.continue_training and (args.model_path is None or not Path(args.model_path).exists()):
        print("NO MODEL TO CONTINUE TRAINING!!!")
        print("{0} does NOT EXISTS!".format(args.model_path))
        sys.exit(1)

    if args.tpath is not None:
        train_data_loc = args.tpath
    if args.vpath is not None:
        validation_data_loc = args.vpath
    if args.glove_path is not None:
        glove_file = args.glove_path

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
    
    print("all_stories.shape=", np.array(all_stories).shape)
    print("all_labels.shape=", np.array(all_labels).shape)
    print("all_data.shape=", all_data.shape)

    strategy = EnsembleStrategy(Evaluator(), args.spath, args.use_gpu, glove_file, args.continue_training, args.model_path)
    #strategy = TopicDiscoveryStrategy(TopicDiscoveryEvaluator(), save_path, args.use_gpu, glove_file, args.continue_trainining)
    if not args.use_ensemble:
        #strategy = TopicDiscoveryStrategy(TopicDiscoveryEvaluator())
        #strategy = SentimentTrajectoryStrategy(SentimentTrajectoryEvaluator())
        #strategy = NBStrategy(PerDataPointEvaluator())
        #strategy = StylisticFeaturesStrategy(OnlyValidationDataEvaluator())
        strategy = LanguageModelStrategy(Evaluator(), args.spath, args.use_gpu, glove_file, args.continue_training, args.model_path)
        # strategy = LSTMClassifierStrategy(Evaluator(), args.spath, args.use_gpu, glove_file, args.continue_training, args.model_path)
        #strategy = TopicConsistencyStrategy(Evaluator(), args.use_gpu)
    validation_error = strategy.evaluator.validation_error(strategy, all_data, validation_data)

    #validation_error = Evaluator.validation_error(strategy, train_data, validation_data)
    print('\nValidation error: {}'.format(validation_error))
