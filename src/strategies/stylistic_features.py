import numpy as np

from sklearn.model_selection import train_test_split
from strategies import Strategy
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.naive_bayes import MultinomialNB as MB

class StylisticFeaturesStrategy(Strategy):
    def fit(self, data: np.ndarray) -> None:
        #This one is special it expects the validation set :/
        
        self.correct_label = 1
        self.wrong_label = 0
        
        #To check how nice this actually is!
        train_stories, test_stories = train_test_split(data, train_size = 0.9)
       
        #Decompose the data 
        IDs = train_stories[:,0]
        partial_stories = train_stories[:,1:5]
        endings = train_stories[:,5:7]
        correct = train_stories[:,7].astype(int)
        
        #Complete the data set
        stories, labels = self.complete_dataset(partial_stories, endings, correct)
        
        #Extract features
        stories = self.extract_features(stories, fit=True)
         
        #Start counting
        self.classifier = MB()
        self.classifier.fit(stories, labels)
        
        # TEMP
        pred = self.predict(test_stories)
        test_labels = test_stories[:,7].astype(int) # Last column has labels
        print(np.mean(np.equal(pred, test_labels)))
        # TEMP
        
        pass

    def predict(self, data: np.ndarray) -> str:
        #Decompose the data
        IDs = data[:,0]
        partial_stories = data[:,1:5]
        endings = data[:,5:7]
        
        # [n_samples, n_blocks-1] --> [2*n_samples, n_blocks]
        stories, _ = self.complete_dataset(partial_stories, endings)
        stories_features = self.extract_features(stories)
        
        # From here assume that stories[:,i:i+1] is the same story with diff endings
        # as longs as i is an even number! not optimal but cba to make it nice for now
        
        predictions = self.classifier.predict_proba(stories_features)
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
        stories = np.apply_along_axis(lambda x: ' '.join(x), 1, stories)
        
        if fit:
            #Fit when asked to
            self.fitted_extracters = True
            self.word_vectorizer = CV(analyzer='word',ngram_range=(1,2))
            self.word_vectorizer.fit(stories)
        elif not self.fitted_extracters:
            raise ValueError("Cannot extract features before fitting,"
                + "run extract_features with fit=True at least once before"
                +   " extracting features.")
        
        #Only words for now
        return self.word_vectorizer.transform(stories)
        
    def complete_dataset(self, partial_stories, endings, correct_endings=None):
        # partial_stories       [n_samples, n_sentences-1] (ending missing)
        # endings               [n_samples, n_options]
        # correct_endings       [n_samples] (values in range [1, n_options])
                
        #No bullshit
        if len(endings) != len(partial_stories):
            raise ValueError("The amount of stories given is not the same"
                                + " as the amount of correct endings")
        
        n_samples, n_options = endings.shape
        
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
        
        """
        #TODO: maybe fix this mess ?
        labels = []
        expanded = []        
        for ind, partial in enumerate(partial_stories):
            current_options = endings[ind]  #list of size 2
            current_correct_ind = int(correct_endings[ind]) - 1 #number indicating correct
            current_wrong_ind   = (current_correct_ind + 1) % 2 # maps 1->0 and 0->1
            
            correct_ending = current_options[current_correct_ind]
            wrong_ending   = current_options[current_wrong_ind]
            
            complete_correct = np.append(partial, correct_ending)
            complete_wrong   = np.append(partial, wrong_ending)
            
            expanded.append(complete_correct)
            expanded.append(complete_wrong)
            
            #For training, while testing you don't know this ofc
            if correct_endings != None:
                labels.extend([0,1])
                        
        return np.array(expanded), np.array(labels)
        """