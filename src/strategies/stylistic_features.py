import numpy as np
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from strategies import Strategy
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.svm import SVC as SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.decomposition import TruncatedSVD as PCA

class StylisticFeaturesStrategy(Strategy):
    def fit(self, data: np.ndarray) -> None:
        #This one is special it expects the validation set :/
        
        #np.random.seed()
        # Parameters
        self.word_gram_range = (1,4)
        self.char_gram_range = (1,5)
        self.correct_label = 1
        self.wrong_label = 0
        self.perform_PCA = False
        self.n_components_PCA = 200
        self.classifier = MB()
        self.train_split = 0.7
        
        # To check how nice this ACTUALLY is!
        train_stories, test_stories = train_test_split(data, train_size = self.train_split)
       
        # Decompose the data 
        _ = train_stories[:,0]                      # Some shit I don't know
        _ = train_stories[:,1:5]                    # Story leading up to options
        endings = train_stories[:,5:7]              # 2 options
        correct = train_stories[:,7].astype(int)    # indicates correct option
                
        labels = self.complete_labels(correct, 2)   # Create the labels for all the options
        
        # Extract features of ONLY THE ENDINGS!!!
        ending_feats = self.extract_features(endings.flatten(), fit=True)
         
        #Start fitting to see if we capture anything that isn't just noise!
        self.classifier.fit(ending_feats, labels)
        
        # TEMP
        pred = self.predict(test_stories)
        test_labels = test_stories[:,7].astype(int) # Last column has labels
        print("Actual validation accuracy: {}".format( np.mean(np.equal(pred, test_labels))))
        # TEMP
        
        pass

    def predict(self, data: np.ndarray) -> str:
        
        # Decompose the data
        _ = data[:,0]
        _ = data[:,1:5]
        endings = data[:,5:7]
        
        # Extract features of ONLY THE ENDINGS!!!
        endings_features = self.extract_features(endings.flatten())
        
        # --   From here it's a bit hacky but it can be cleaned if needed    --
        # From here assume that stories[:,i:i+1] is the same story with diff endings
        # as longs as i is an even number! not optimal but cba to make it nice for now
        
        predictions = self.classifier.predict_proba(endings_features)
        predictions = np.reshape(predictions, (-1,2,2)) #[n_samples, 2, 2]
        predictions = [self.resolve_prediction(distr_for_options) for distr_for_options in predictions]
                
        return np.array(predictions)

    def resolve_prediction(self, distributions_per_options):
        # distributions_per_options     [n_options, n_classes=2]
        # Here we decide what happens, if both options predict the same
        # answer, then we assign that answer to the one with highest probability
        # If they predict different results there is no problem!
        # --
        # Method should return an integer denoting the index of the CORRECT sentence
        # according to predictions...
        
        # [n_options, 2] --> [n_options] contains which class is 
        # preferred by each option.
        decision = np.argmax(distributions_per_options, axis=1)
        
        if self.all_equal(decision):
            # They all think they are wrong/correct, however only one can be!
            decision_for_both = decision[0] # Do they think they are wrong/correct?
            
            # How confident are they in their respective decision?
            priority = np.argmax(distributions_per_options, axis = 0)
            
            # Find the option who is most confident in it's decision
            final = priority[decision_for_both]
            
            # That's the final decision
            return final + 1
        else:
            return np.argmax(decision) + 1 
            # If the index is i then it's the i+1'th sentence
            # They preferred different classes (One of them thinks it's a wrong ending
            # and one of them thinks it's a correct ending)
        
    def all_equal(self, array):        
        return np.unique(array).size == 1
    
    def extract_features(self, stories, fit=False):        
        if fit:
            self.fit_extractors(stories)
        elif not self.fitted_extracters:
            raise ValueError("Cannot extract features before fitting,"
                + "run extract_features with fit=True at least once before"
                +   " extracting features.")
        
        feats = [extr.transform((stories)) for extr in self.get_extractors()]
        full_features = sp.hstack(tuple(feats))
        
        # Doesn't worrk too well since variance is high in any direction
        # because the bag of words/ngrams is pretty sparse (i think)
        if self.perform_PCA:
            print("Performing PCA, this might take a while!")
            full_features = PCA(n_components = self.n_components_PCA).fit_transform(full_features)
        
        return full_features
        
        
    def fit_extractors(self, stories):
        self.fitted_extracters = True
        self.word_vectorizer = CV(analyzer='word',ngram_range=self.word_gram_range) # Word grams
        self.char_vectorizer = CV(analyzer='char',ngram_range=self.char_gram_range,min_df=5) # Char
        self.length_exctractor = LengthExtractor()
        
        # One loop to fit them all and in darkness ... fit them?
        for extractor in self.get_extractors():
            extractor.fit(stories)
            
    def get_extractors(self):
        # Put the feature extractors here!
        return [
                self.word_vectorizer, 
                self.char_vectorizer, 
                self.length_exctractor
                ]
        
    def complete_dataset(self, partial_stories, endings, correct_endings=None):
        # partial_stories       [n_samples, n_sentences-1] (ending missing)
        # endings               [n_samples, n_options]
        # correct_endings       [n_samples] (values in range [1, n_options])
                
        #No bullshit
        if len(endings) != len(partial_stories):
            raise ValueError("The amount of stories given is not the same"
                                + " as the amount of correct endings")
        
        _, n_options = endings.shape
        
        complete_stories = self.complete_stories(partial_stories, endings)
        labels = self.complete_labels(correct_endings, n_options)
        
        return complete_stories, labels
        
    def complete_stories(self, partial_stories, endings):
        # Check above for dimensions
        
        #No bullshit
        if len(partial_stories) != len(endings):
            raise ValueError( "Not every story has a (happy) end!"
                            + " Or equally sad not every ending has a happy story :(")
        
        expanded_set = []
        for (part, endings) in zip(partial_stories, endings):
            completed_stories = [np.append(part, ending) for ending in endings]
            expanded_set.extend(completed_stories)
            
        return np.array(expanded_set)
        
    def complete_labels(self, correct_endings, n_options):
        #Check above for dimensions and meanings
        
        if correct_endings is None:
            return None
        
        #Assume value "i" means sentence "i" which means INDEX "i-1"
        correct_endings = correct_endings - 1
        
        expanded_labels = []
        for correct_ending in correct_endings:
            labels = [self.wrong_label]*n_options
            labels[correct_ending] = self.correct_label
            expanded_labels.extend(labels)
        
        return np.array(expanded_labels)

# Custom extracters go here they should have at least fit() and transform()
class LengthExtractor:
    def fit(self, data):
        pass
    
    def transform(self, data):
        # data is [n_samples] values are documents
        # It's to transpose the array so it's viewed as a matrix so as to be able to hstack it later!
        return sp.csr_matrix(np.vectorize(len)(data)[:,np.newaxis])
        