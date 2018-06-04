import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.svm import SVC

from strategies import Strategy
from extractors.sentiment_trajectory_extractor import SentimentTrajectoryExtractor
from extractors.embedded_closeness_extractor import EmbeddedClosenessExtractor
from sklearn.linear_model import LogisticRegression as lr
class EnsembleStrategy(Strategy):
        
    # Expects an Augmented training set.
    def fit(self, train: np.ndarray, val: np.ndarray, aug: np.ndarray) -> None:
        sentiment = SentimentTrajectoryExtractor()
        sentiment.fit(train)
        
        embed_close = EmbeddedClosenessExtractor()
        embed_close.fit(train)
        
        #expanded_validation_x, expanded_validation_labels = self.expand_validation(val)
        #print(expanded_validation_x.shape)
        #expanded_validation_x = np.column_stack((np.zeros((len(expanded_validation_x), 1)), expanded_validation_x))
        #print(expanded_validation_x.shape)
        
        
        feat_2 = sentiment.extract(expanded_validation_x)
        feat_1 = embed_close.extract(expanded_validation_x)
        
        #print(feat_1)
        
        X = np.column_stack((feat_2, feat_1))
        
        #stuff = np.reshape(stuff, (-1, 2))
        #choice = np.argmax(stuff, axis=1) + 1
        
        #print(choice)
        
        #val_labels = val[:,-1].astype(int)
        
        #print(np.average(np.equal(val_labels, choice)))
        
        classifier = lr()
        classifier.fit(X, expanded_validation_labels)
        
        #Only correct proba
        #distr_for_endings = np.array(classifier.predict_proba(X))[:,1]
        #print(distr_for_endings)
        #distr_for_endings = np.reshape(distr_for_endings, (-1,2))
        
        #choice = np.argmax(distr_for_endings, 1) + 1
        
        #print(np.average(np.equal(val_labels, choice)))
        
        pass

    def predict(self, data: np.ndarray) -> str:
        return None

        
    def expand_validation(self, validation):
        expanded_x = []
        expanded_y = []
        
        for val in validation:
            story = val[:5]
            e_1 = val[5]
            e_2 = val[6]
            label = val[7]
            expanded_x.append(np.append(story, e_1))
            expanded_x.append(np.append(story, e_2))
            labs = [0]*2
            labs[int(label) -1] = 1
            expanded_y.extend(labs)
        
        return np.array(expanded_x), np.array(expanded_y)