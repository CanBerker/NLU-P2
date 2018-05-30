from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from strategies import Strategy

import nltk
import random
import zipfile
import numpy as np


class NBStrategy(Strategy):

    def __init__(self, evaluator):
        nltk.download('stopwords')

        self.evaluator = evaluator
        self.data_path = "../data/reviews"
        with zipfile.ZipFile("{0}/reviews.zip".format(self.data_path), 'r') as zip_ref:
            zip_ref.extractall("{0}/.".format(self.data_path))
        self.word_gram_range = (2,3)
        self.diff_sentiment = 0
        self.test_sz = 0.2

    def fit(self, data: np.ndarray) -> None:
        with open("{0}/reviews.txt".format(self.data_path)) as f:
            reviews = f.read().split("\n")
        with open("{0}/labels.txt".format(self.data_path)) as f:
            labels = f.read().split("\n")
        # tokenizing
        reviews_tokens = [review.split() for review in reviews]
        stop = stopwords.words('english')
        self.count_vectorizer = CountVectorizer(analyzer=u'word', ngram_range=self.word_gram_range, stop_words=stop)
        reviews_counted = self.count_vectorizer.fit_transform(reviews)
        self.tfidf_transformer = TfidfTransformer(use_idf=True)
        tfidf_comments = self.tfidf_transformer.fit_transform(reviews_counted)
        X_train, X_test, y_train, y_test = train_test_split(tfidf_comments, labels, test_size=self.test_sz, random_state=None)
        # model creation
        self.bnbc = BernoulliNB(binarize=None)
        self.bnbc.fit(X_train, y_train)

        score = self.bnbc.score(X_test, y_test)
        print("BernoulliNB Training score:{0}".format(score))

    def predict(self, data: np.ndarray) -> str:
        # read stories
        stories = []
        answers1 = []
        answers2 = []
        real_choice = []

        stories.append(",".join(data[1:4]))
        answers1.append(data[5])
        answers2.append(data[6])
        real_choice.append(data[7])

        # sentiment analysis of stories
        stories_counted = self.count_vectorizer.transform(stories)
        tfidf_stories = self.tfidf_transformer.transform(stories_counted)
        stories_predicted = self.bnbc.predict(tfidf_stories)
        # sentiment analysis of answers_1
        a1_counted = self.count_vectorizer.transform(answers1)
        tfidf_a1 = self.tfidf_transformer.transform(a1_counted)
        a1_predicted = self.bnbc.predict(tfidf_a1)
        # sentiment analysis of answers_2
        a2_counted = self.count_vectorizer.transform(answers2)
        tfidf_a2 = self.tfidf_transformer.transform(a2_counted)
        a2_predicted = self.bnbc.predict(tfidf_a2)

        if (stories_predicted[0] == a1_predicted[0]):
            return '1'
        elif (stories_predicted[0] == a2_predicted[0]):
            return '2'
        else:
            # if we didn't predict the same sentiment as story then <pad>
            self.diff_sentiment += 1
            return str(random.randint(1, 2))