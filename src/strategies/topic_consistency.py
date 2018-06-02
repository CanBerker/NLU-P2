import numpy as np
import math

from strategies import Strategy
from collections import Counter

from random import randint
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

class TopicConsistencyStrategy(Strategy):
    def fit(self, data: np.ndarray) -> None:
        self.log("There is no training for this.")
        pass


    def remove_uninteresting_tags(self, tagged_text):
        # http://nishutayaltech.blogspot.com/2015/02/penn-treebank-pos-tags-in-natural.html
        interesting_types = ['RB', 'NNP', 'NNS', 'NN', 'VBD', 'VBG', 'VBN', 'VB', 'VBZ', 'PRP$', 'POS', 'MD', 'PRP', 'JJ', 'CD', 'CC', 'DT', 'FW']
        filtered_tags_text = []
        for story_tags in tagged_text:
            filtered_tags = [story_tag for story_tag in story_tags if story_tag[1] in interesting_types]
            filtered_tags_text.append(filtered_tags)
        return filtered_tags_text

    def merge_sentences(self, data):
        #data:      [n_stories, n_sentences]
        #return:    [n_stories]
        return np.apply_along_axis(lambda x: ' '.join(x), 1, data)

    def predict(self, data: np.ndarray) -> str:
        self.log("Predicting on actual dataset")
        partial = data[:,1:5]
        endings = data[:,5:]

        full_stories = data[:,2:7]

        stories_words = self.merge_sentences(partial)

        tmp_stories_tags = [pos_tag(word_tokenize(story)) for story in stories_words]
        self.stories_pos_tags = self.remove_uninteresting_tags(tmp_stories_tags)

        tmp_endings_pos_tags = [[pos_tag(word_tokenize(ending[0])), pos_tag(word_tokenize(ending[1]))] for ending in endings]
        self.endings_pos_tags = [self.remove_uninteresting_tags(ending) for ending in tmp_endings_pos_tags]
        predictions = []
        for i in range(len(self.stories_pos_tags)):
            tagged_story = self.stories_pos_tags[i]
            endings = self.endings_pos_tags[i]
            sim_e1 = self.get_pos_tag_sim(tagged_story, endings[0])
            sim_e2 = self.get_pos_tag_sim(tagged_story, endings[1])
            choice = randint(1, 2)
            if sim_e1 != sim_e2:
                choice = 1 if sim_e1 > sim_e2 else 2
            else:
                self.log("Endings from Story={0} were found similar".format(i))
            predictions.append(choice)
        return predictions

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