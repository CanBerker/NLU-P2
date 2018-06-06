import numpy as np
import math
import time

from extractors import Extractor
from collections import Counter

from random import randint
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from utils.loader import load_glove
from scipy.spatial.distance import cosine


class EmbeddedClosenessExtractor(Extractor):
    def __init__(self, embedding_path="glove.6B.50d.txt"):
        self.embedding_path = embedding_path

    def fit(self, data: np.ndarray) -> None:
        

        self.word_representation = self.define_representation(data)
        
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
        
    def extract(self, data: np.ndarray) -> str:
        self.log("Predicting on actual dataset")
        _ = data[:,:2]
        partial = data[:,2:6]
        endings = data[:,6]

        full_stories = data[:,2:7]
        
        stories_words = self.merge_sentences(partial)

        print("--Tagging {} stories--".format(len(data)))
        self.stories_pos_tags = self.pos_tag_data(stories_words)
        self.stories_pos_tags = self.remove_uninteresting_tags(self.stories_pos_tags)
        print("--Done tagging stories--\n")
        
        print("--Tagging {} endings--".format(len(endings)))
        pos_tagged_endings = self.pos_tag_data(endings)
        pos_tagged_endings = self.remove_uninteresting_tags(pos_tagged_endings)
        print("--Done tagging endings--\n")
                
        closenesses = self.find_closeness_per_story(self.stories_pos_tags, pos_tagged_endings)
        
        self.endings_pos_tags = pos_tagged_endings
        
        return closenesses[:,np.newaxis]

    def find_closeness_per_story(self, stories, endings):
        #stories array of [[pos_tags]] of shape (n_samples, ??)
        #for now endings is a [[pos_tags]] of shape (n_samples, ??)        
        return np.array([ self.ending_story_closeness(story, ending) for 
                                (story, ending) in list(zip(stories, endings))])
    
    def ending_story_closeness(self, story, ending):
        #Defines a closeness metric between story and ending.
        
        #unable to define a distance if no words are in ending??
        if len(ending) == 0:
            return 0
            
        cosine_per_tag = [self.max_embedding_closeness(story, end_tag) for end_tag in ending]
        
        return np.average(cosine_per_tag)
        
    def max_embedding_closeness(self, story, end_tag):
    
        # unable to find a distance if there are no words in the story
        if len(story)==0:
            return 0
            
        return np.amax([self.word_distance(end_tag, story_tag) for story_tag in story])
    
    def word_distance(self, word_1, word_2):
        embedding_1 = self.resolve(word_1[0], self.word_representation)
        embedding_2 = self.resolve(word_2[0], self.word_representation)
        #close = 1/ (np.linalg.norm(embedding_1 - embedding_2) +1)
        close = 1 - cosine(embedding_1, embedding_2)
        if math.isnan(close):
            close = 0
            
        #print("Closeness of {} and {}:          {}".format(word_1[0], word_2[0], close))
        return close
        
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