import numpy as np
import nltk
import time
from sklearn.model_selection import train_test_split

def calculate_classification_stats(prediction, actual, f_lab = -1, t_lab = 1):
    if len(prediction) != len(actual):
        raise ValueError('Prediction must have thhe same length as actual results!')
    
    res = [find_case(x, f_lab, t_lab) for x in zip(prediction, actual)]
    
    counts = {'tp':0, 'fn':0, 'tn':0, 'fp':0}
    for x in res:
        counts[x] +=1
    
    return counts['tp'],counts['fn'],counts['tn'],counts['fp']
    
def find_case(tuple, f_lab, t_lab):
    pr, ac = tuple
    if ac == f_lab and pr == ac:
        return 'tn'
    elif ac == f_lab and pr != ac:
        return 'fp'
    elif ac == t_lab and pr == ac:
        return 'tp'
    elif ac == t_lab and pr != ac:
        return 'fn'
        
def write_to_log(str):
    with open("log/log.txt", "a") as file:
        file.write(str)
        
def reset_log():
    open("log/log.txt", 'w').close()
    
def is_number(x):
    try:
        float(x)
        return True
    except:
        return False

def file_exists(path):
    return Path(path).is_file()
    
def inverse_map(mp):
    return {v: k for k, v in mp.items()}
    
def convert_to_int(word, word_to_int):
    global total, unconverted
    total += 1
    try:
        return word_to_int[word]
    except:       
        unconverted += 1
        return word_to_int['<unk>']

total = 0
unconverted = 0

def embed_to_ints(X, word_to_int):
    global total, unconverted
    start = time.time()
    print("--starting to embed--")
    embedded = []
    for sentence in X:
        embedded_s = [convert_to_int(w, word_to_int) for w in sentence]
        embedded.append(embedded_s)
    end = time.time()
    print("--done embedding-- {0}".format(end-start))
    print("---->Words not embedded: {0}%".format(unconverted/total))
    return embedded
    
def get_batches(X, y, batch_size):
    n_batches = len(X)//batch_size
    left = True if len(X) % batch_size > 0 else False
    
    batches = []
    for i in range(n_batches):
        x_batch = X[i*batch_size:(i+1)*batch_size]
        y_batch = y[i*batch_size:(i+1)*batch_size]
        batches.append((x_batch, y_batch))
    #if left:
    #    batches.append((X[n_batches*batch_size:], y[n_batches*batch_size:]))
    
    return batches

def get_test_batches(X, batch_size):
    n_batches = len(X)//batch_size
    left = True if len(X) % batch_size > 0 else False
    
    batches = []
    for i in range(n_batches):
        x_batch = X[i*batch_size:(i+1)*batch_size]
        batches.append(x_batch)
    if left:
        batches.append(X[n_batches*batch_size:])
    
    return batches
    
def print_tweet_statistics(*tweet_sets):
    for tweet_set in tweet_sets:        
        print_statistics(tweet_set)
        
def print_statistics(tweets):
    n_tw, max_seq = np.array(tweets).shape
    length_per_tweet = np.count_nonzero(tweets, axis=1)
    avg_tw_length = np.mean(length_per_tweet)
    largest_emb = np.amax(tweets)
    lowest_emb = np.amin(tweets)
    avg_index = np.mean(tweets)
    median_index = np.median(tweets)
    
    print("Total amount of tweets:  {0}".format(n_tw))
    print("Maximum sequence length: {0}".format(max_seq))
    print("Longest tweet present:   {0}".format(np.amax(length_per_tweet)))
    print("Shortest tweet present:  {0}".format(np.amin(length_per_tweet)))
    print("Average tweet length:    {0}".format(avg_tw_length))
    print("Largest embedding:       {0}".format(largest_emb))
    print("Smallest embedding:      {0}".format(lowest_emb))
    print("Average index:           {0}".format(avg_index))
    print("Median index:            {0}".format(median_index))
    
def pad_data(X, max_seq_size):
    start = time.time()
    print("--starting to pad data--")
    for x in X:
        if len(x) > max_seq_size:
            print("Line {0} already has length of {1}".format(x, len(x)))
            
    padded_data = [ np.pad(x, (max_seq_size - len(x)%max_seq_size, 0), 'constant') for x in X]
    end = time.time()
    print("--Done padding data -- {0}\n".format(end-start))
    return np.array(padded_data)
    
def split(X,y,train_size=0.9):
    return train_test_split(X, y, train_size)