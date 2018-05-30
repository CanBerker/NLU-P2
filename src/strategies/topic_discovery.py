import numpy as np
import scipy.sparse as sp
import time
import _pickle as cPickle 

from scipy.spatial.distance import cosine as cos_sim

from strategies import Strategy

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.svm import SVC as SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.decomposition import TruncatedSVD as PCA
from sklearn.decomposition import LatentDirichletAllocation as LDA

class TopicDiscoveryStrategy(Strategy):
    def fit(self, data: np.ndarray, val: np.ndarray) -> None:
        # This one is special it expects the validation set :/
        
        #np.random.seed()
        # Parameters 
        self.ngram_range = (1,1)
        self.n_topics = 1
        self.topic_extractor = LDA(n_components = self.n_topics, evaluate_every=0, n_jobs = -1, max_iter = 1)
        
        #Decompose data
        _ = data[:,0]
        _ = data[:,1]
        partial_stories = data[:,2:6]
        endings = data[:,6]
                
        bag_of_words = self.extract_bag_of_words(partial_stories, fit=True)
        
        start = time.time()
        print("Starting to fit latent topic discovery, this might take a while.")
        self.topic_extractor.fit(bag_of_words)
        with open('my_dumped_classifier.pkl', 'wb') as fid:
            cPickle.dump(self.topic_extractor, fid)   
        print("Finished LTD... {}".format(time.time() - start))
        
        pass

    def predict(self, data: np.ndarray) -> str:
        
        # Decompose the data
        _ = data[:,0]
        partial_stories = data[:,1:5]
        endings = data[:,5:7]
        
        partial_stories = self.extract_bag_of_words(partial_stories)        
        endings = self.extract_bag_of_words(endings.flatten()[:,np.newaxis])
        
        print(partial_stories.shape, endings.shape)
        
        topic_distribution_part = self.topic_extractor.transform(partial_stories)
        topic_distribution_ending = self.topic_extractor.transform(endings)
        # [n_samples*2, n_topics]
        
        # [n_samples*2, n_topics] --> [n_samples, 2, n_topics]
        topic_distribution_endings = np.reshape(topic_distribution_ending, (-1, 2, self.n_topics))
        
        decisions = []
        for (story_distr, endings_distr) in list(zip(topic_distribution_part, topic_distribution_endings)):
            decisions.append(np.argmin([cos_sim(story_distr, ending_distr) for ending_distr in endings_distr]))
        
        # {0,1} --> {1,2}
        decisions = np.array(decisions) + 1
        
        return decisions
    
    def extract_bag_of_words(self, stories, fit=False):
        # stories       [n_stories, n_sentences]        
        
        print(stories.shape)
        
        stories = np.apply_along_axis(lambda x: ' '.join(x), 1, stories)
        
        if fit:
            self.is_fitted = True
            self.vectorizer = CV(ngram_range=self.ngram_range) # Vanilla BOW
            print("Fitting vectorizer on {}".format(stories.shape))
            self.vectorizer.fit(stories)
        elif not self.is_fitted:
            raise ValueError("Cannot extract BOW before fitting,"
                +   " run extract_bag_of_words with fit=True at least once before"
                +   " extracting features.")
        
        print("Starting to transform data.")
        return self.vectorizer.transform(stories)

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
        