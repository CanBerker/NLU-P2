import math
import math
import time

import numpy as np
import tensorflow as tf
import keras

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer as tk
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import nltk

from strategies import Strategy
from utils.loader import load_glove
from utils.utils import convert_to_int, embed_to_ints


class LanguageModelStrategy(Strategy):

    def fit(self, data: np.ndarray) -> None:
        self.max_vocab = 5000
        self.oov_token = "<unk>"
        self.embedding_size = 100
        self.hidden_size = 100
        self.use_dropout = True
        self.dropout_rate = 0.5
        self.train_size = 0.85
        self.optimizer = Adam()
        self.num_epochs = 1
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()
        
        
        # Decompose data
        _ = data[:,0]
        _ = data[:,1]
        _ = data[:,2:6]
        _ = data[:,6]
        
        # Get full stories to train language model on.
        full_stories = data[:,2:7]
        
        # Paste sentences next to each other
        stories_strings = self.merge_sentences(full_stories)
        stories_strings = self.clean_strings([lambda x: x.lower(),
                                              lambda x: x.replace(".", " . ")],stories_strings)
        
        # Tokenize the data into tokens using the standard tokenizer
        stories_tokenized = self.tokenize_data(self.tokenizer, stories_strings)
        
        # Load the embeddings of choice.
        word_to_emb, word_to_int, int_to_emb = load_glove(self.glove_path)
        
        train_x, validation_x = train_test_split(stories_tokenized, train_size = self.train_size)
        
        # Reduce the embeddings to what you need only.
        word_to_emb, word_to_int, int_to_emb = self.reduce_embedding(int_to_emb, word_to_int, word_to_emb, train_x)
        
        # Embed our tokens with the reduced embedding
        train_embedded = embed_to_ints(train_x, word_to_int=word_to_int)
        valid_embedded = embed_to_ints(validation_x, word_to_int=word_to_int)
        
        print("Example embedding:{}".format(self.int_to_words(self.inverse_map(word_to_int),valid_embedded)))
        
        # Build the LSTM LM with softmax
        embedding_matrix = np.array(int_to_emb)
        self.max_vocab, _ = embedding_matrix.shape
        model = self.build_graph(embedding_matrix)
        
        train_generator = KerasBatchGenerator(train_embedded,
                                              self.max_vocab,
                                              skip_step=5)
        valid_data_generator = KerasBatchGenerator(valid_embedded,
                                                   self.max_vocab,
                                                   skip_step=5)

        checkpointer = ModelCheckpoint(filepath=self.save_path + '/model-{epoch:02d}.hdf5', verbose=1)

        model.fit_generator(train_generator.generate(),
                             steps_per_epoch=train_generator.n_batches,#len(train_x) // (self.batch_size * self.max_seq_size),
                             epochs=self.num_epochs,
                             validation_data=valid_data_generator.generate(),
                             validation_steps=valid_data_generator.n_batches,
                             callbacks=[checkpointer]
                             )
                             
        self.test_model(valid_embedded, self.inverse_map(word_to_int))

    def reduce_embedding(self, int_to_emb, word_to_int, word_to_emb, tokens):
        
        # All next to each other
        tokens = np.concatenate(tokens)
        
        # Find unique tokens and count them
        unsorted_uniques, unsorted_counts = np.unique(tokens, return_counts = True)
        
        # print Some stuff
        print("Average token frequency:{}".format(np.average(unsorted_counts)))
        print("Total amount of unique tokens:{}".format(len(unsorted_uniques)))
        
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
        
        print("Total amount of tokens attempted:      {}".format(self.total_tried))
        print("Total amount of tokens failed to embed:{}".format(self.total_failed))
        print("---> {}%".format(self.total_failed/self.total_tried))
        print("Shape of new embedding:{}".format(np.array(r_int_to_emb).shape))
        
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
        
    def test_model(self, train_data, reversed_dictionary):
        model = load_model(self.save_path + "/model-{}.hdf5".format(str(self.num_epochs).zfill(2)))
        dummy_iters = 40
        example_training_generator = KerasBatchGenerator(train_data, self.max_vocab, skip_step=1)
        # print("Training data:")
        # for i in range(dummy_iters):
        #     dummy = next(example_training_generator.generate())
        num_predict = 10
        true_print_out = "Actual words: "
        pred_print_out = "Predicted words: "
        for i in range(num_predict):
            data = next(example_training_generator.generate())
            prediction = model.predict_proba(data[0])
            predict_word = np.argmax(prediction,2)
            print("true_print_out=", self.int_to_words(reversed_dictionary, data[0]))
            print("pred_print_out=", self.int_to_words(reversed_dictionary, predict_word))
            # true_print_out += reversed_dictionary[data[0][i]] + " "
            # pred_print_out += reversed_dictionary[predict_word] + " "


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
            self.log("Using GPU!")
            config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
            sess = tf.Session(config=config)
            keras.backend.set_session(sess)
        vocab_size, embed_size = embedding_matrix.shape
        vocab_size, embed_size = embedding_matrix.shape
        model = Sequential()
        model.add(Embedding(vocab_size, embed_size, weights=[embedding_matrix], trainable=False))
        model.add(LSTM(self.hidden_size, return_sequences=True))
        #model.add(LSTM(self.hidden_size, return_sequences=True))
        if self.use_dropout:
            model.add(Dropout(self.dropout_rate))
        model.add(Dense(vocab_size))
        model.add(Activation('softmax'))
        
        model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer, metrics=['sparse_categorical_accuracy'])
        print(model.summary())
        
        return model

    def predict(self, data: np.ndarray) -> str:
        return None

    def merge_sentences(self, data):
        #data:      [n_stories, n_sentences]
        #return:    [n_stories]
        return np.apply_along_axis(lambda x: ' '.join(x), 1, data)
        
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

    def __init__(self, data, vocabulary, skip_step=5):
        self.data = data
        #self.num_steps = num_steps
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step
        self.grouped_by_length = list(self.group_by_length(self.data).items())
        self.preferred_batch_size = 64
        #self.preferred_batch_size = 32
        self.n_batches = np.sum([math.ceil(len(v)/self.preferred_batch_size) for l,v in self.grouped_by_length])

    def group_by_length(self, data):
        dict = {}
        for d in data:
            if len(d) not in dict:
                dict[len(d)] = []
            dict[len(d)].append(d)
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
        data = np.array(data)
        num_batches = math.ceil(len(data)/pref_batch_size)
        batches = []
        for i in range(num_batches):
            data_slice = data[i*pref_batch_size: (i+1)*pref_batch_size]
            x_slice = data_slice[:,:-1]
            y_slice = data_slice[:,1:]
            #y_slice = to_categorical(y_slice, num_classes=self.vocabulary)

            batches.append((x_slice, np.expand_dims(y_slice, -1)))

        return batches
