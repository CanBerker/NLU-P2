import numpy as np
import random
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from strategies import Strategy

class SentimentTracjectoryStrategy(Strategy):
    def fit(self, data: np.ndarray) -> None:
    
        self.sentimentAnalyzer = SentimentIntensityAnalyzer()
        self.story_grouping = (1,2,1,1) #Assume beginning -> body -> climax -> ending
        self.n_values = 3   #Negative, Neutral and Positive
        self.done = 0 #sanity
        
        
        #Decompose the data 
        IDs = data[:,0]
        titles = data[:,1]
        stories = data[:,2:]
        
        #Group sentences
        stories = self.group_stories(stories, self.story_grouping)
        
        #Convert the stories to sentiment trajectories [n_samples, n_groups]
        #Values contain 0, 1, 2 for neg, neutral and positive sentiment
        trajectories = self.find_trajectories(stories)
        print(trajectories.shape)
        trajectories = np.sign(trajectories) + 1 #+1 to shift from [-1,1] to [0,2]
        print(trajectories.shape)
        
        #Returns a [n_values, ..., n_values] where len(...) = len(story_grouping)
        counts = self.count_elements(trajectories, self.n_values,
                                                smoothing = 5)
        pass

    def predict(self, data: np.ndarray) -> str:
        return '1'
    
    def count_elements(self, objs, n_values, smoothing=1):
        #Method assumes that values are in range [0,n_values] (n_values excluded)
        n_objs, n_steps = objs.shape
        shape = (n_values,)*n_steps
        
        counts = np.full(shape, smoothing)
        
        for obj in objs:
            counts[tuple(obj)] +=1
            
        print(counts)
        return counts
        
    def find_trajectory(self, story):
        
        sa = self.sentimentAnalyzer
        for sentence in story:
            sentiment = [0]*len(story)
            #sentiment = sa.polarity_scores(sentence)['compound']
            
        #Sanity
        self.done +=1
        if self.done % 1000 == 0:
            print("Done {0}".format(self.done))
        #Sanity
        
        return sentiment
        
    def find_trajectories(self, stories):
        return np.array([self.find_trajectory(story) for story in stories])
        
    def group_stories(self, data, grouping, merge_fn = lambda x: ' '.join(x)):
        print(data.shape)
        n_samples, n_feat = data.shape
        n_groups = len(grouping)
        
        if n_feat != np.sum(grouping):
            raise ValueError('Cannot group, expected {0} sentences per sample but received {1}'.format(np.sum(grouping), n_feat))
        
        
        grouped_data = []
        low_ind = 0
        for column_i, group_size in enumerate(grouping):
            column = data[:,low_ind:low_ind + group_size]
            column = [merge_fn(row) for row in column]
            grouped_data.append(column)
            low_ind += group_size
        
        grouped_data = np.array(grouped_data).T
        print(grouped_data[0])
        return grouped_data
            