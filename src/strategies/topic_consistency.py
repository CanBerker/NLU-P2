import numpy as np
import math
import time

from strategies import Strategy
from collections import Counter

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from utils.loader import load_glove
from scipy.spatial.distance import cosine

class TopicConsistencyStrategy(Strategy):
    def fit(self, data: np.ndarray) -> None:
        
        #self.embedding_path = "glove.6B.50d.txt"
        self.embedding_path = self.glove_path
        self.word_representation = self.define_representation(data)
    
        self.log("There is no training for this.")
        pass

    def define_representation(self, data):
        word_to_emb, _, _ = load_glove(self.embedding_path)
        return word_to_emb

    def remove_uninteresting_tags(self, tagged_text):
        # http://nishutayaltech.blogspot.com/2015/02/penn-treebank-pos-tags-in-natural.html
        interesting_types = ['JJ', 'JJR', 'JJS', 'NN', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        filtered_tags_text = []
        for story_tags in tagged_text:
            filtered_tags = [story_tag for story_tag in story_tags if story_tag[1] in interesting_types]
            filtered_tags_text.append(filtered_tags)
        return filtered_tags_text

    def merge_sentences(self, data):
        #data:      [n_stories, n_sentences]
        #return:    [n_stories]
        return np.apply_along_axis(lambda x: ' '.join(x), 1, data)

    def pos_tag_data(self, strings_list):
        return [pos_tag(word_tokenize(string)) for string in strings_list]
        
    def predict(self, data: np.ndarray) -> str:
        self.log("Predicting on actual dataset")
        partial = data[:,1:5]
        endings = data[:,5:]

        full_stories = data[:,2:7]

        stories_words = self.merge_sentences(partial)

        print("--Tagging {} stories--".format(len(data)))
        self.stories_pos_tags = self.pos_tag_data(stories_words)
        self.stories_pos_tags = self.remove_uninteresting_tags(self.stories_pos_tags)
        print("--Done tagging stories--\n")
        
        print("--Tagging {} endings--".format(len(data)*2))
        pos_tagged_endings = self.pos_tag_data(endings.flatten())
        pos_tagged_endings = self.remove_uninteresting_tags(pos_tagged_endings)
        pos_tagged_endings = np.reshape(np.array(pos_tagged_endings), (-1,2))
        print("--Done tagging endings--\n")
        
        self.endings_pos_tags = pos_tagged_endings
        predictions = []
        
        print("--Starting to compute distances--")
        start = time.time()
        for i, tagged_story in enumerate(self.stories_pos_tags):
            tagged_endings = self.endings_pos_tags[i]
            
            avg_closeness = []
            for tagged_end in tagged_endings:
                avg_closeness.append(self.get_embedded_closeness(tagged_end, tagged_story))
            
            choice = np.argmax(avg_closeness) + 1
            predictions.append(choice)
            
            if i %100 ==0:
                print("Done {}".format(i))
        print("--Done computing distances--{}".format(time.time() - start))
        return predictions

    def resolve(self, word, word_to_emb):
        try:
            return word_to_emb[word.lower()]
        except:
            return word_to_emb["<unk>"]
        
    def get_embedded_closeness(self, ending_tags, story_tags):
        word_to_emb = self.word_representation
        
        if len(ending_tags) == 0:
            return 0
        if len(story_tags) == 0:
            return 0
            
        embedded_story  = [self.resolve(word, word_to_emb) for (word, tag) in story_tags]
        embedded_ending = [self.resolve(word, word_to_emb) for (word, tag) in ending_tags]
        
        closeness_mat = []
        for end_word_emb in embedded_ending:
            closeness_arr = []
            for story_word_emb in embedded_story:
                close = 1 - cosine(end_word_emb, story_word_emb)
                closeness_arr.append(close)
            closeness_mat.append(closeness_arr)
        
        closeness_mat = np.array(closeness_mat)
        max_close = np.amax(closeness_mat, axis=1)
        
        #for word_e, tag_e in ending_tags:
        #    for word_s, tag_s in story_tags:
        #        e_emb = self.resolve(word_e, word_to_emb)
        #        s_emb = self.resolve(word_s, word_to_emb)
        #        dist = 1 - cosine(s_emb, e_emb)
        #        print("closeness between: {} and {} = {}".format(word_e, word_s, dist))
                
        return np.average(max_close)
        
    def get_pos_tag_sim(self, tagged_story, ending):
        accum_sim = 0.0
        non_tagged_story = Counter([t[0].lower() for t in tagged_story])
        non_tagged_ending = Counter([t[0].lower() for t in ending])
        intersection = set(non_tagged_story.keys()) & set(non_tagged_ending.keys())
        numerator = sum([non_tagged_story[x] * non_tagged_ending[x] for x in intersection])
        sum1 = sum([non_tagged_story[x]**2 for x in non_tagged_story.keys()])
        sum2 = sum([non_tagged_ending[x]**2 for x in non_tagged_ending.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        cos_sim = 0.0
        if denominator:
           cos_sim = float(numerator) / denominator
        return cos_sim
