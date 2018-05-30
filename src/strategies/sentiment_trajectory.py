import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier as rf

from strategies import Strategy

class SentimentTrajectoryStrategy(Strategy):
    def __init__(self, evaluator):
        import nltk
        nltk.download('vader_lexicon')
        self.evaluator = evaluator

    def fit(self, data: np.ndarray) -> None:
    
        self.sentimentAnalyzer = SentimentIntensityAnalyzer()
        self.story_grouping = (1,2,1,1) #Assume beginning -> body -> climax -> ending
        self.n_values = 3   #Negative, Neutral and Positive
        self.done = 0 #sanity
        self.n_sentences = np.sum(self.story_grouping) #How many sentences to consider?
        
        #Decompose the data 
        IDs = data[:,0]
        titles = data[:,1]
        stories = data[:,-self.n_sentences:]        
        
        self.classifier = rf()
        
        #Group sentences to form blocks for sentiment
        stories = self.group_stories(stories, self.story_grouping)
        
        #Convert the stories to sentiment trajectories [n_samples, n_groups]
        #Values contain 0, 1, 2 for neg, neutral and positive sentiment
        trajectories = self.find_trajectories(stories)
                
        #___ Counting classifier       
        #Returns a [n_values, ..., n_values] where len(...) = len(story_grouping)
        self.counts = self.count_elements(trajectories, self.n_values, smoothing = 5)
        #___ SVC classifier
        #self.classifier.fit(trajectories[:,:-1], trajectories[:,-1])
        #___
        
        pass

    def predict(self, data: np.ndarray) -> str:
        #Decompose data
        #--> data[:,0] contains ID'sa
        #--> data[:,1-5] contains first 4 sentences
        #--> data[:,5-7] contains 2 ending options
        IDs = data[:,0]
        partial_stories = data[:,5-(self.n_sentences-1):5]
        endings = data[:,5:7]
        
        # IMPORTANT: group test set exectly the same as when training otherwise 
        # nothing makes sense! Except for the last sentence which by definition
        # of test set cannot be in the grouping
        partial_stories = self.group_stories(partial_stories,
                                    self.story_grouping[:-1])
        
        sentiment_for_endings = self.find_trajectories(endings)
        sentiment_distributions = self.find_ending_distribution(partial_stories)
        
        endings_probabilities = self.map_probabilities(sentiment_for_endings, sentiment_distributions)
                
        # +1 to select sentence
        return np.argmax(endings_probabilities, axis=1) + 1

    def map_probabilities(self, indices, distribution):
        #The one-liner to rule them all
        return [distr[indices] for (indices, distr) in list(zip(indices, distribution))]
    
    def find_ending_distribution(self, partial_stories):
        partial_trajectories = self.find_trajectories(partial_stories)
        return [self.counts[tuple(partial_trajectory)] for partial_trajectory in partial_trajectories]
        #return self.classifier.predict_proba(partial_trajectories)
        
    def count_elements(self, objs, n_values, smoothing=1):
        #Method assumes that values are in range [0,n_values] (n_values excluded)
        n_objs, n_steps = objs.shape
        shape = (n_values,)*n_steps
        
        #Smoothing changes nothing at this moment (count not used as prior)
        counts = np.full(shape, smoothing)
        
        for obj in objs:
            counts[tuple(obj)] +=1
            
        return counts
    
    def find_sentiment(self, sentence):
        #negative = -1
        #neutral  =  0
        #positive =  1
        sa = self.sentimentAnalyzer
        
        ##negative = 0
        #neutral  =  1
        #positive =  2
        return 1+int(np.sign(sa.polarity_scores(sentence)['compound']))
        
    def find_trajectory(self, story):
    
        trajectory = [self.find_sentiment(part) for part in story]
        #trajectory = [0]*len(story)
        
        #Sanity
        self.done +=1
        if self.done % 1000 == 0:
            print("Done {0}".format(self.done))
        #Sanity
        
        return trajectory
        
    def find_trajectories(self, stories):
        return np.array([self.find_trajectory(story) for story in stories])
        
    def group_stories(self, data, grouping, merge_fn = lambda x: ' '.join(x)):
        n_samples, n_feat = data.shape
        n_groups = len(grouping)
        
        if n_feat != np.sum(grouping):
            raise ValueError('Cannot group, expected {0} sentences per sample but received {1}'.format(np.sum(grouping), n_feat))

        #TODO: make nicer
        grouped_data = []
        low_ind = 0
        for group_size in grouping:
            column = data[:,low_ind:low_ind + group_size]
            column = [merge_fn(row) for row in column]
            grouped_data.append(column)
            low_ind += group_size
        
        return np.array(grouped_data).T