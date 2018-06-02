import numpy as np
import pickle
import time

from pathlib import Path
from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import csv

def generate_indices(tweets):
    return np.arange(1, len(tweets)+1)
    
def load_vocabulary(vocab_file):
    return pickle.load(open(vocab_file, 'rb'))

def get_labels(tweets, label):
    return [label]*len(tweets)

def join_and_shuffle(tweet_collections, tweet_labels):
    if len(tweet_collections) != len(tweet_labels):
        raise ValueError("Given tweet collection and labels are not the same size! Check join_and_shuffle(.) call")
        
    for i in range(len(tweet_collections)):
        if len(tweet_collections[i]) != len(tweet_labels[i]):
            raise ValueError('One of the collections has mismatched label amount')
    
    print("--Starting to shuffle--")
    all_tw = []
    all_la = []
    for i, list in enumerate(tweet_collections):
        all_tw.extend(list)
        all_la.extend(tweet_labels[i])
    
    shuffled_tw, shuffled_la = shuffle(all_tw, all_la)
    print("--Done shuffling--\n")
    return shuffled_tw, shuffled_la

# Help function to clean some text if you want
def clean_lines(lines, fns=[lambda x:x]):
    print("cleaning lines")
    lines_c = []
    for i, line in enumerate(lines):
        line_c = line
        for fn in fns:
            line_c = fn(line_c)
        lines_c.append(line_c)
    print("Done cleaning {0} lines".format(len(lines_c)))
    return lines_c

# DEBUG FUNCTION FOR FASTER LOADING OF EMBEDDING/TOKENIZATIONS
def load_embedded_tweets(tweet_file, data_dir):
    path = "{0}/embedded_data/{1}.npy".format(data_dir, tweet_file)
    start = time.time()
    print("-- starting to load embedded tweets--")
    embedding = np.load(path)
    end = time.time()
    print("--Done loading embedded tweets -- {0} \n".format(end-start))
    return embedding
    
def save_embedded_tweets(tweet_file, data_dir, X):
    path = "{0}/embedded_data/{1}".format(data_dir, tweet_file)
    np.save(path, X)
    
#Loads a collection of files of tweets in raw form (lines)
def load_tweets(*tweet_files):
    start = time.time()
    print("--Starting to load tweets--")
    tweets_per_file = []
    for tweet_file in tweet_files:
        tweets = open(tweet_file, encoding="utf8")
        tweets_per_file.append(tweets.readlines())
        
    end = time.time()
    print("--End loading tweets-- {0} \n".format(end-start))
    return tweets_per_file
    
def preprocess(vocab, pos_data, neg_data, test_data):
    return vocab, pos_data, neg_data, test_data

def load_glove(file_path):
    start = time.time()
    words_loaded = 0
    
    print ("--Loading embedding: {0} --".format(file_path))
    f = open(file_path,'r', encoding='utf8')
    model = {}
    int_to_emb = []
    word_to_int = {}
    for i, line in enumerate(f):
        
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        
        if word in word_to_int:
            print("Word {0} has been found multiple times".format(word))
        
        #word_to_embed
        model[word] = embedding
        #word to int i.e. position in matrix
        word_to_int[word] = i
        #int to emb i.e. matrix
        int_to_emb.append(embedding)
        
        words_loaded += 1
        
    end = time.time()
    print ("--Done loading embedding-- {0} \n---->{1} words loaded \n".
                        format(end-start, words_loaded))
    
    return model, word_to_int, int_to_emb
    
def load_embedding(file_path, format=None):
    if format=='glove':
        return load_glove(file_path)
    else:
        return None

def pad_data(X, max_seq_size):
    start = time.time()
    print("--starting to pad data--")
    for x in X:
        if len(x) > max_seq_size:
            print("Line {0} already has length of {1}".format(x, len(x)))
            
    padded_data = [ np.pad(x, (0, max_seq_size - len(x)%max_seq_size), 'constant') for x in X]
    end = time.time()
    print("--Done padding data -- {0}\n".format(end-start))
    return np.array(padded_data)


def save_prediction(predictions, f_dir, f_name):
    preds = np.round(predictions, decimals=0).reshape(-1).astype(int)
    preds[preds == 0] = -1

    with open(f_dir + f_name + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Id', 'Prediction'])
        for row in enumerate(preds, 1):
            writer.writerow(row)
