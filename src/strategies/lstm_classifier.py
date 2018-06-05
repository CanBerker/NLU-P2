import math
import time

import numpy as np
import tensorflow as tf
import keras
import nltk

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer as tk
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from strategies import Strategy
from utils.loader import load_glove
from utils.utils import convert_to_int, embed_to_ints


class LSTMClassifierStrategy(Strategy):

    def fit(self, data: np.ndarray) -> None:
        self.oov_token = "<unk>"
        self.embedding_size = 100
        self.hidden_size = 64
        self.use_dropout = True
        self.train_size = 0.8
        self.dropout_rate = 0.5
        self.optimizer = Adam()
        self.num_epochs = 20
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.glove_path = "glove.6B.50d.txt"
        # Decompose data
        _ = data[:,0]
        _ = data[:,1]
        _ = data[:,2:6]
        _ = data[:,6]
        
        # Get full stories to train language model on.
        full_stories = data[:,2:7]
        labels = data[:,7]
        
        # Paste sentences next to each other
        stories_strings = self.merge_sentences(full_stories)
        stories_strings = self.clean_strings([lambda x: x.lower(),
                                              lambda x: x.replace(".", " . ")],stories_strings)

        # Tokenize the data into tokens using the standard tokenizer
        stories_tokenized = self.tokenize_data(self.tokenizer, stories_strings)
        # Load the embeddings of choice.
        word_to_emb, word_to_int, int_to_emb = load_glove(self.glove_path)
        # separate training and validation data
        train_x, validation_x, train_lab, validation_lab = train_test_split(stories_tokenized,
                                                                            labels, train_size = self.train_size)

        self.log("Amount of training samples: {}".format(len(train_x)))
        self.log("Amount of testing samples: {}".format(len(validation_x)))
        
        # Reduce the embeddings to what you need only.
        self.word_to_emb, self.word_to_int, self.int_to_emb = self.reduce_embedding(int_to_emb, word_to_int, word_to_emb, train_x)

        # Embed our tokens with the reduced embedding
        train_embedded = embed_to_ints(train_x, word_to_int=self.word_to_int)
        valid_embedded = embed_to_ints(validation_x, word_to_int=self.word_to_int)

        # Build the LSTM LM with softmax
        embedding_matrix = np.array(self.int_to_emb)
        self.max_vocab, _ = embedding_matrix.shape
        self.model = self.build_graph(embedding_matrix)

        # Define data generators
        train_generator = KerasBatchGenerator(train_embedded, train_lab)
        valid_data_generator = KerasBatchGenerator(valid_embedded, validation_lab)

        model_save_name = '/model-{epoch:02d}.hdf5'
        if self.continue_training:
            self.log("Loading model from {}".format(self.model_path))
            self.model = load_model(self.model_path)
            model_save_name = '/model-cont-{epoch:02d}.hdf5'

        checkpointer = ModelCheckpoint(filepath=self.save_path + model_save_name, verbose=1)

        self.model.fit_generator(train_generator.generate(),
                             steps_per_epoch=train_generator.n_batches,#len(train_x) // (self.batch_size * self.max_seq_size),
                             epochs=self.num_epochs,
                             validation_data=valid_data_generator.generate(),
                             validation_steps=valid_data_generator.n_batches,#len(validation_x)//(self.batch_size) ,#len(validation_x)//(self.batch_size * self.max_seq_size),
                             callbacks=[checkpointer]
                             )


    def reduce_embedding(self, int_to_emb, word_to_int, word_to_emb, tokens):
        #Takes in an embedding and takes out only what it needs (i.e. tokens)
        self.log("-- Reducing embedding--")
        # All next to each other
        tokens = np.concatenate(tokens)
        
        # Find unique tokens and count them
        unsorted_uniques, unsorted_counts = np.unique(tokens, return_counts = True)
        
        # print Some stuff
        self.log("Average token frequency:{}".format(np.average(unsorted_counts)))
        self.log("Total amount of unique tokens:{}".format(len(unsorted_uniques)))
        
        # Sort tokens by frequency
        sorted_unique_tokens = list(zip(unsorted_uniques, unsorted_counts))
        sorted_unique_tokens.sort(key=lambda t: t[1], reverse=True)
        sorted_unique_tokens = sorted_unique_tokens[:self.max_vocab-1]
        
        _, emb_dim = np.array(int_to_emb).shape
        
        #For statistics
        self.total_tried = 0
        self.total_failed= 0
        
        r_word_to_int   = {w:i                                     for i, (w,c) in enumerate(sorted_unique_tokens)}
        r_int_to_emb    = [self.resolve(w, word_to_emb, emb_dim)   for i, (w,c) in enumerate(sorted_unique_tokens)]
        r_word_to_embed = {w:self.resolve(w, word_to_emb, emb_dim) for i, (w,c) in enumerate(sorted_unique_tokens)}
        
        r_word_to_int[self.oov_token] = len(sorted_unique_tokens)
        r_word_to_embed[self.oov_token] = self.resolve(self.oov_token, word_to_emb, emb_dim)
        r_int_to_emb.append(self.resolve(self.oov_token, word_to_emb, emb_dim))

        self.log("Total amount of tokens attempted:      {}".format(self.total_tried))
        self.log("Total amount of tokens failed to embed:{}".format(self.total_failed))
        self.log("---> {}%".format(self.total_failed/self.total_tried))
        self.log("Shape of new embedding:{}".format(np.array(r_int_to_emb).shape))

        self.log("--Done reducing the embeddings--\n")
        return r_word_to_embed, r_word_to_int, r_int_to_emb
    
    def resolve(self, word, word_to_emb, emb_dim):
        self.total_tried += 1
        try:
            emb = word_to_emb[word]
        except:
            emb = np.random.rand(emb_dim)
            #print("Was not able to find an embedding for:{}".format(word))
            self.total_failed +=1
        return emb
        
    def clean_strings(self, cleaners, data):
        for cleaner in cleaners:
            data = np.vectorize(cleaner)(data)
        return data
        
    def tokenize_data(self, tokenizer, data):
        #data:      [n_stories]
        start = time.time()
        print("--Starting to tokenize--")
        res = np.array([tokenizer.tokenize(string) for string in data])
        print("--Done tokenizing--{}\n".format(time.time()-start))
        return res
        
    def test_model(self, valid_data, labels):
        model = load_model(self.save_path + "/model-{}.hdf5".format(str(self.num_epochs).zfill(2)))
        example_training_generator = KerasBatchGenerator(valid_data, labels)
        
        gen = example_training_generator.generate()
        
        preds = []
        actuals = []
        sum = 0
        total_samples = len(valid_data)
        for i in range(example_training_generator.n_batches):
            (batch_x, batch_y) = next(gen)
            pred_b = model.predict(batch_x)
            preds.extend(pred_b)
            actuals.extend(batch_y)
            
        print(actuals, preds)
        
        print("Final test acc:{}".format(np.mean(np.equal(pred_b, actuals.astype(int)))))


    def int_to_words(self, reversed_dictionary, data):
        # data[batch_size, num_steps]

        all_samples = []
        for sample in data:
            s = []
            for i in sample:
                s.append(reversed_dictionary[i])
            all_samples.append(' '.join(s))

        return np.array(all_samples)

    def build_graph(self, embedding_matrix):
        self.log("Building graph")
        if (self.use_gpu):
            config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
            sess = tf.Session(config=config)
            keras.backend.set_session(sess)
        vocab_size, embed_size = embedding_matrix.shape
        model = Sequential()
        model.add(Embedding(vocab_size, embed_size, weights=[embedding_matrix], trainable=False))
        model.add(LSTM(self.hidden_size,return_sequences=True))
        model.add(LSTM(self.hidden_size))
        if self.use_dropout:
            model.add(Dropout(self.dropout_rate))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['acc'])
        print(model.summary())
        
        return model

    def predict(self, data: np.ndarray) -> str:
        #Decompose data
        #--> data[:,0] contains ID'sa
        #--> data[:,1-5] contains first 4 sentences
        #--> data[:,5-7] contains 2 ending options
        
        print(self.model.summary())
        choices = []
        for partial_story in data:
            partial = partial_story[1:5]
            endings = partial_story[5:7]
                        
            full_stories = [np.append(partial, end) for end in endings]
            
            full_stories = self.merge_sentences(full_stories)            
            
            full_stories = self.clean_strings([lambda x: x.lower(),
                                              lambda x: x.replace(".", " . ")],full_stories)
            
            full_stories = self.tokenize_data(self.tokenizer, full_stories)
            full_embed = np.array(embed_to_ints(full_stories, self.word_to_int))
            
            
            predictions = []
            for end in full_embed:
                predictions.append(self.model.predict(np.array([end]))[0])
                print(self.int_to_words(self.inverse_map(self.word_to_int), [end]))
            
            
            choice = np.argmax(predictions) + 1
            print(predictions, choice)
            choices.append(choice)
            
        return choices

    def merge_sentences(self, data):
        #data:      [n_stories, n_sentences]
        #return:    [n_stories]
        return np.array([' '.join(x) for x in data])
        
    def fit_tokenizer(self, stories):
        #stories:   [n_stories]
        self.tokenizer = tk(self.max_vocab, oov_token = self.oov_token)
        self.tokenizer.fit_on_texts(stories)        
        
     
    def get_int_mappings(self):
        
        word_to_int  = self.tokenizer.word_index
        int_to_word  = self.inverse_map(word_to_int)
        return word_to_int, int_to_word
        
    def inverse_map(self, map):
        return {v: k for k, v in map.items()}
        
    def embed_data(self, text_data, word_to_int):
        #return [word_to_int[w] for w in text_data if w in word_to_int ]
        return self.tokenizer.texts_to_sequences(text_data)
        
    def pad_data(self, X, max_seq_size):
        start = time.time()
        self.log("--starting to pad data--")
        for x in X:
            if len(x) > max_seq_size:
                print("Line {0} already has length of {1}".format(x, len(x)))
        
        padded_data = []
        for x in X:
            pad_n = max_seq_size - len(x)%max_seq_size if len(x) < max_seq_size else 0
            padded_data.append(np.pad(x, (pad_n, 0), 'constant'))
            
        end = time.time()
        self.log("--Done padding data -- {0}\n".format(end-start))
        return np.array(padded_data)
        
    def find_max_seq(self, data):
        #data:      [n_samples, None] (None is variable length)
        max_len = -1
        min_len = 10000
        for i, s in enumerate(data):
            s_len = len(s)
            if s_len > max_len:
                max_len = s_len
            if s_len < min_len:
                min_len = s_len
                _, int_to_word = self.get_int_mappings()
                print(i, "=", [int_to_word[ss] for ss in s])
                
        return [max_len, min_len]

class KerasBatchGenerator(object):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        
        # A list of tuples [(length, [(sample_of_size_length, label)])]
        self.grouped_by_length = list(self.group_by_length(self.data, self.labels).items())
        self.preferred_batch_size =  42
        
        self.n_batches = np.sum([math.ceil(len(samples)/self.preferred_batch_size) for l, samples in self.grouped_by_length])
        print("Number of batches found:{}".format(self.n_batches))

    def group_by_length(self, data, labels):
        #Gets a dict that maps: length --> collection or tuples (sample, label)
        dict = {}
        if len(data) != len(labels):
            self.log("ERROR: More data than labels. len(data)={0} len(labels)={1}".format(len(data), len(labels)))

        for i in range(len(data)):
            sample = data[i]
            lab = labels[i]
            if len(sample) not in dict:
                dict[len(sample)] = []
            dict[len(sample)].append((sample,lab))
            
        return dict

    def generate(self):

        # print(np.average([len(x)/self.batch_size for l, x in grouped_by_length.items()]))
        while True:
            for i in range(len(self.grouped_by_length)):
                lg, l_data = self.grouped_by_length[i]
                batches = self.createBatch(l_data, self.preferred_batch_size)
                for b in batches:
                    yield b[0], b[1]


    def createBatch(self, data, pref_batch_size):
        #data is a list of [(sample, label)] where every sample is of same size len(sample)
        samples_of_size, labels = zip(*data)
        samples_of_size = np.array(samples_of_size)
        labels = np.array(labels)
        
        #print("Sample shape :{} label shape:{}".format(samples_of_size.shape, labels.shape))
        
        num_batches = math.ceil(len(data)/pref_batch_size)
        
        batches = []
        for i in range(num_batches):
            x_batch = samples_of_size[i*pref_batch_size: (i+1)*pref_batch_size]
            y_batch = labels[i*pref_batch_size: (i+1)*pref_batch_size]
            #y_batch = to_categorical(y_batch, num_classes=2)

            batches.append((np.array(x_batch), np.array(y_batch)))

        return batches
