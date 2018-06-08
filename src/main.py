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
    test_data_loc = os.path.join(os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'), 'test.csv')
    eth_data_loc = os.path.join(os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'), 'test_nlu18_utf-8.csv')

    glove_file = "glove.6B.50d.txt"
    save_path = "checkpoint"
    ensemble_approaches = ["SentimentTrajectory"]

    parser = ArgumentParser()
    parser.add_argument("-t", "--training_dataset", dest='tpath', help="Training dataset path")
    parser.add_argument("-v", "--validation_dataset", dest='vpath', help="Validation dataset path")
    parser.add_argument("-s", "--save_model_path", dest='spath', default=save_path, help="Model directory")
    parser.add_argument("-m", "--model_to_train", dest='model', help="Model to be trained")
    parser.add_argument("-g", "--use_gpu", dest='use_gpu', action='store_true', help="Use GPU for training")
    parser.add_argument("-gp", "--glove_path", dest='glove_path', help="Glove file path")
    parser.add_argument("-en", "--use_ensemble", dest='use_ensemble', action='store_true',
                        help="Use ensemble of methods")
    parser.add_argument('-ea', "--ensemble_approaches", dest="ensemble_approaches", type=str, nargs='*',
                        help='Ensemble approaches (none defaults to SentimentTrajectory)')
    parser.add_argument("-vt", "--validate_on_testset", dest='validate_on_test', action='store_true', default=False,
                        help="Perform validation on test set")
    parser.add_argument("-eth", "--validate_on_eth_set", dest='validate_on_eth', action='store_true', default=False,
                        help="Perform validation on ETH test set")
    parser.add_argument("-ct", "--continue_training", dest='continue_training', action='store_true', default=False,
                        help="Continue training")
    parser.add_argument("-mp", "--model_cont_training_path", dest='model_path', default=None,
                        help="Model path to continue training")
    parser.add_argument("-lc_mp", "--lstm_class_model_path", dest='lstm_class_model_path', default=None,
                        help="LSTM classifier model path to be used for ensembling.")
    parser.add_argument("-lm_mp", "--lang_model_path", dest='lang_model_model_path', default=None,
                        help="Language model model path to be used for ensembling.")

    args = parser.parse_args()
    print(args)

    if args.validate_on_test:
        print("Validating on TEST SET. File={}".format(test_data_loc))
        validation_data_loc = test_data_loc
    if args.validate_on_eth:
        print("Validating on TEST SET. File={}".format(eth_data_loc))
        validation_data_loc = eth_data_loc

    if args.ensemble_approaches is not None:
        ensemble_approaches = args.ensemble_approaches

    if not Path(args.spath).is_dir():
        print("THERE IS NO SAVE DIRECTORY!!!")
        print("{0} does NOT EXISTS!".format(args.spath))
        sys.exit(1)

    if args.continue_training and (args.model_path is None or not Path(args.model_path).exists()):
        print("NO MODEL TO CONTINUE TRAINING!!!")
        print("{0} does NOT EXISTS!".format(args.model_path))
        sys.exit(1)
    if args.use_ensemble:
        if args.lstm_class_model_path is None or not Path(args.lstm_class_model_path).exists():
            print("NO LSTM CLASSIFIER MODEL SUPPLIED!!!")
            print("{0} does NOT EXISTS!".format(args.lstm_class_model_path))
            sys.exit(1)
        if args.lang_model_model_path is None or not Path(args.lang_model_model_path).exists():
            print("NO LANGUAGE MODEL SUPPLIED!!!")
            print("{0} does NOT EXISTS!".format(args.lang_model_model_path))
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

    if args.validate_on_eth:
        n_samples, n_sentences = validation_data.shape
        extra_col = np.zeros((n_samples, 1))
        validation_data = np.column_stack((extra_col, validation_data, extra_col))#Make same format as other validation
        print(validation_data.shape)

    negative_samples = augment_data(train_data, 1, load=False)
    positive_samples = train_data
    
    negative_lables = get_labels(negative_samples, 0)
    positive_labels = get_labels(positive_samples, 1)
    
    all_stories, all_labels = join_and_shuffle([negative_samples, positive_samples],
                                            [negative_lables, positive_labels])
    
    aug_data = np.column_stack((all_stories, all_labels))
    print("all_stories.shape=", np.array(all_stories).shape)
    print("all_labels.shape=", np.array(all_labels).shape)
    print("aug_data.shape=", aug_data.shape)

    strategy = EnsembleStrategy(EnsembleEvaluator(train_data, validation_data, aug_data, args.validate_on_eth),
                                args.spath, args.use_gpu, args.lstm_class_model_path, args.lang_model_model_path,
                                ensemble_approaches, glove_file)

    if not args.use_ensemble:
        #strategy = TopicDiscoveryStrategy(TopicDiscoveryEvaluator(validation_data))
        #strategy = SentimentTrajectoryStrategy(SentimentTrajectoryEvaluator())
        #strategy = NBStrategy(PerDataPointEvaluator(train_data, validation_data))
        #strategy = StylisticFeaturesStrategy(OnlyValidationDataEvaluator())
        regular_eval = Evaluator(aug_data, validation_data)
        #strategy = LanguageModelStrategy(regular_eval, args.spath, args.use_gpu, glove_file, args.continue_training, args.model_path)
        strategy = LSTMClassifierStrategy(regular_eval, args.spath, args.use_gpu, glove_file, args.continue_training, args.model_path)
        #strategy = TopicConsistencyStrategy(regular_eval, args.use_gpu)
        
    validation_error = strategy.evaluator.validation_error(strategy)
    print('\nValidation error: {}'.format(validation_error))
