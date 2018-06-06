import numpy as np
import sys

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.svm import SVC

from strategies import Strategy

from extractors.sentiment_trajectory_extractor import SentimentTrajectoryExtractor
from extractors.embedded_closeness_extractor import EmbeddedClosenessExtractor
from extractors.lstm_classifier_extractor import LSTMClassifierExtractor
#from extractors.sentence_embedding_extractor import SentenceEmbeddingExtractor

from sklearn.linear_model import LogisticRegression as lr
class EnsembleStrategy(Strategy):

    # Expects an Augmented training set.
    def fit(self, train: np.ndarray, val: np.ndarray, aug: np.ndarray) -> None:
        
        self.log(aug.shape)

        self.extractors = []
        sentiment = SentimentTrajectoryExtractor()
        sentiment.fit(train)
        self.extractors.append(sentiment)
        
        embed_close = EmbeddedClosenessExtractor()
        embed_close.fit(train)
        self.extractors.append(embed_close)
        
        lstm_c = LSTMClassifierExtractor(self.glove_path, self.save_path)
        lstm_c.fit(aug)
        self.extractors.append(lstm_c)
        
        # self.sentence_emb = SentenceEmbeddingExtractor()
        # self.sentence_emb.fit(aug)
        
        stories = aug[:,:7]
        labels  = aug[:,-1]
        
        features = self.extract_features(stories)
        
        self.classifier = lr()
        self.classifier.fit(features, labels)
        
        pass

    def predict(self, val: np.ndarray) -> str:   
        
        # [n_samples, [e1, e2]] -> [2*n_samples, ei]
        expanded_validation_x = self.expand_validation(val)

        feats = self.extract_features(expanded_validation_x)
        
        probs = self.classifier.predict_proba(feats)[:,1]
        probs = np.reshape(probs, (-1, 2))
        
        #Find the most confident ending
        predictions = np.argmax(probs, axis=1) + 1 #Index thing
        
        return predictions
        
    def extract_features(self, data):
        #Data must be [n_samples, 7]
        feats = [extr.extract(data) for extr in self.extractors]
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