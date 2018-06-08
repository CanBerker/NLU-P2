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

        # How are the words represented? (e.g. embeddings)
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
        #Decompose data
        #--> data[:,0] contains ID'sa
        #--> data[:,1] contains title
        #--> data[:,2-6] contains first 4 sentences
        #--> data[:,6] contains ending
        #--> data[:,7] contains labels
        self.log("Predicting on actual dataset")
        _ = data[:,:2]
        partial = data[:,2:6]
        endings = data[:,6]

        full_stories = data[:,2:7]
        
        # Paste all stories next to each other
        stories_words = self.merge_sentences(partial)

        # POS tag all the stories
        self.log("--Tagging {} stories--".format(len(data)))
        self.stories_pos_tags = self.pos_tag_data(stories_words)
        self.stories_pos_tags = self.remove_uninteresting_tags(self.stories_pos_tags)
        self.log("--Done tagging stories--\n")
        
        # Pos tag all the endings
        self.log("--Tagging {} endings--".format(len(endings)))
        pos_tagged_endings = self.pos_tag_data(endings)
        pos_tagged_endings = self.remove_uninteresting_tags(pos_tagged_endings)
        self.log("--Done tagging endings--\n")
        
        # Find embedded closeness value per story.
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
        
        #Find max cosine distance per ending word.
        cosine_per_tag = [self.max_embedding_closeness(story, end_tag) for end_tag in ending]
        
        return np.average(cosine_per_tag)
        
    def max_embedding_closeness(self, story, end_tag):
    
        # unable to find a distance if there are no words in the story
        if len(story)==0:
            return 0
            
        return np.amax([self.word_distance(end_tag, story_tag) for story_tag in story])
    
    def word_distance(self, word_1, word_2):
        # Distancce
        embedding_1 = self.resolve(word_1[0], self.word_representation)
        embedding_2 = self.resolve(word_2[0], self.word_representation)
        close = 1 - cosine(embedding_1, embedding_2)
        if math.isnan(close):
            close = 0
            
        return close
        
    def resolve(self, word, word_to_emb):
        try:
            return word_to_emb[word.lower()]
        except:
            return word_to_emb["<unk>"]
        