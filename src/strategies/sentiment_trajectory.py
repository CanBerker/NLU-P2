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
        #trajectories = np.sign(trajectories).astype(int) + 1 #+1 to shift from [-1,1] to [0,2]
                
        #Returns a [n_values, ..., n_values] where len(...) = len(story_grouping)
        self.counts = self.count_elements(trajectories, self.n_values, smoothing = 5)
        
        print(self.counts)
        
        pass

    def predict(self, data: np.ndarray) -> str:
        IDs = data[:,0]
        partial_stories = data[:,1:5]
        endings = data[:,5:7]
        
        # IMPORTANT: group test set exectly the same as when training otherwise 
        # nothing makes sense! Except for the last sentence which by definition
        partial_stories = self.group_stories(partial_stories, self.story_grouping[:-1])
        
        
        actual_endings_for_options = self.find_trajectories(endings)
        most_likely_endings = self.find_most_likely_endings(partial_stories)
        
        distance = np.abs((actual_endings_for_options.T - most_likely_endings).T)
        
        same = 0
        for d in distance:
            if d[0] == d[1]:
                same += 1
        print(same, len(data))
        
        closest = np.argmin(distance, axis=1) + 1
        
        return closest

    def find_most_likely_endings(self, partial_stories):
        partial_trajectories = self.find_trajectories(partial_stories)
        #partial_trajectories = np.sign(partial_trajectories).astype(int) + 1
        
        print(partial_trajectories)
        max = np.apply_along_axis(lambda x: np.argmax(self.counts[tuple(x)]),
                                                        1, partial_trajectories)
        print(max)
        return max
        
    def count_elements(self, objs, n_values, smoothing=1):
        #Method assumes that values are in range [0,n_values] (n_values excluded)
        n_objs, n_steps = objs.shape
        shape = (n_values,)*n_steps
        
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
            print(trajectory)
        #Sanity
        
        return trajectory
        
    def find_trajectories(self, stories):
        return np.array([self.find_trajectory(story) for story in stories])
        
    def group_stories(self, data, grouping, merge_fn = lambda x: ' '.join(x)):
        n_samples, n_feat = data.shape
        n_groups = len(grouping)
        
        if n_feat != np.sum(grouping):
            raise ValueError('Cannot group, expected {0} sentences per sample but received {1}'.format(np.sum(grouping), n_feat))

        grouped_data = []
        low_ind = 0
        for group_size in grouping:
            column = data[:,low_ind:low_ind + group_size]
            column = [merge_fn(row) for row in column]
            grouped_data.append(column)
            low_ind += group_size
        
        return np.array(grouped_data).T
            