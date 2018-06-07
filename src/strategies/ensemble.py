import numpy as np
import time
import sys

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.svm import SVC

from strategies import Strategy

from extractors.sentiment_trajectory_extractor import SentimentTrajectoryExtractor
from extractors.embedded_closeness_extractor import EmbeddedClosenessExtractor
from extractors.lstm_classifier_extractor import LSTMClassifierExtractor
from extractors.language_model_extractor import LanguageModelExtractor
from extractors.sentence_embedding_extractor import SentenceEmbeddingExtractor

from sklearn.linear_model import LogisticRegression as LR


class EnsembleStrategy(Strategy):

    def __init__(self, evaluator, save_path, use_gpu, lstm_class_model_path, lang_model_model_path, approaches, glove_path=None):
        self.extractors = []
        self.use_gpu = use_gpu
        self.evaluator = evaluator
        self.glove_path = glove_path
        self.save_path = save_path
        self.approaches = approaches
        self.lstm_class_model_path = lstm_class_model_path
        self.lang_model_model_path = lang_model_model_path

    # Expects an Augmented training set.
    def fit(self, train: np.ndarray, val: np.ndarray, aug: np.ndarray) -> None:
        #TMP
        _ = val[:,:-1]
        labels = val[:,-1]        
        self.expanded_validation_labels = self.expand_labels(labels)
        #TMP
        
        self.log("augmented_data_shape={}".format(aug.shape))
        self.init_extractors(train, val, aug)        
        self.fit_extractors(self.extractors)

        stories = aug[:,:7]
        labels  = aug[:,-1]
        # feature extraction
        features = self.extract_features(stories)
        # classification using extraction features
        self.classifier = LR()
        self.classifier.fit(features, labels)

    def predict(self, val: np.ndarray) -> str:   
        # [n_samples, [e1, e2]] -> [2*n_samples, ei]
        expanded_validation_x = self.expand_validation(val)

        feats = self.extract_features(expanded_validation_x)
        
        probs = self.classifier.predict_proba(feats)[:,1]
        probs = np.reshape(probs, (-1, 2))
        
        #Find the most confident ending
        predictions = np.argmax(probs, axis=1) + 1 #Index thing
        
        return predictions
    
    def expand_labels(self, labels):
        #labels: list of ints in [1,2]
        all = []
        for lab in labels:
            tmp = [0]*2
            tmp[int(lab) - 1] = 1
            all.extend(tmp)
        
        return all
    def fit_extractors(self, extractors):
        self.log("Fitting extractors")
        for (extr, set) in extractors:
            start = time.time()
            extr.fit(set)
            self.log("Fitting time={} secs".format(time.time() - start))
            
    def init_extractors(self, train, val, aug):
        self.log("Initializing extractors")
        for app in self.approaches:
            self.log("Adding {0} as feature extractor".format(app))
            if app == "SentimentTrajectory":
                self.extractors.append((SentimentTrajectoryExtractor(), train))
            elif app == "EmbeddedCloseness":
                self.extractors.append((EmbeddedClosenessExtractor(self.glove_path), train))
            elif app == "LSTMClassifier":
                self.extractors.append((LSTMClassifierExtractor(self.glove_path, self.lstm_class_model_path), aug))
            elif app == "LanguageModel":
                self.extractors.append((LanguageModelExtractor(self.glove_path, self.lang_model_model_path), aug))
            elif app == "SentenceEmbedding":
                #self.extractors.append((SentenceEmbeddingExtractor("train_embedding.npy","test"), train))
                self.log("Extractor={} not implemented yet!".format(app))
                quit()
        #self.extractors = [
        #                    #(SentimentTrajectoryExtractor(), train),
        #                   #(EmbeddedClosenessExtractor(self.glove_path), train),
        #                   #(LSTMClassifierExtractor(self.glove_path, self.lstm_class_model_path), aug),
        #                   (LanguageModelExtractor(self.glove_path, self.lang_model_model_path), aug),
        #                   #(SentenceEmbeddingExtractor("train_embedding.npy","test"), train),
        #                   (SentenceEmbeddingExtractor("train_embedding_last.npy","valid_embedding_last.npy", self.save_path, self.expanded_validation_labels), train),
        #                   ]
        
    def extract_features(self, data):
        #Data must be [n_samples, 7]
        self.log("Extracting features")
        feats = [extr.extract(data) for (extr, _) in self.extractors]
        return np.column_stack(tuple(feats))
        
    def expand_validation(self, validation):
        expanded_x = []
        
        for val in validation:
            story = val[:5]
            e_1 = val[5]
            e_2 = val[6]
            expanded_x.append(np.append(story, e_1))
            expanded_x.append(np.append(story, e_2))
        
        expanded_stories = np.array(expanded_x)
        titles = np.zeros((len(expanded_stories), 1))
        
        return np.column_stack((titles, expanded_stories))
