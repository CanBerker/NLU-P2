import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.svm import SVC

from strategies import Strategy
from extractors.sentiment_trajectory_extractor import SentimentTrajectoryExtractor
from extractors.embedded_closeness_extractor import EmbeddedClosenessExtractor
from extractors.lstm_classifier_extractor import LSTMClassifierExtractor

from sklearn.linear_model import LogisticRegression as lr
class EnsembleStrategy(Strategy):

    # Expects an Augmented training set.
    def fit(self, train: np.ndarray, val: np.ndarray, aug: np.ndarray) -> None:
        
        print(aug.shape)
        
        self.sentiment = SentimentTrajectoryExtractor()
        self.sentiment.fit(train)
        
        self.embed_close = EmbeddedClosenessExtractor()
        self.embed_close.fit(train)
        
        self.lstm_c = LSTMClassifierExtractor(self.glove_path, self.save_path)
        self.lstm_c.fit(aug)
        
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
        extractors = [
                      self.sentiment,
                      self.embed_close,
                      self.lstm_c,
                      ]
                      
        feats = [extr.extract(data) for extr in extractors]
        
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