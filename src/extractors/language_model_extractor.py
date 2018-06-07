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

from extractors import Extractor
from utils.loader import load_glove
from utils.utils import convert_to_int, embed_to_ints
import sys

class LanguageModelExtractor(Extractor):

    def __init__(self, glove_path, lang_model_model_path):
        self.glove_path = glove_path
        self.lang_model_model_path = lang_model_model_path

    def fit(self, data: np.ndarray) -> None:
        self.max_vocab = 10000
        self.oov_token = "<unk>"
        self.embedding_size = 100
        self.hidden_size = 100
        self.use_dropout = True
        self.dropout_rate = 0.5
        self.train_size = 0.85
        self.optimizer = Adam()
        self.num_epochs = 10
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()
        
        
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
        self.word_to_emb, self.word_to_int, self.int_to_emb = self.reduce_embedding(int_to_emb, word_to_int, word_to_emb, train_x)
        
        # Embed our tokens with the reduced embedding
        # train_embedded = embed_to_ints(train_x, word_to_int=word_to_int)
        # valid_embedded = embed_to_ints(validation_x, word_to_int=word_to_int)
        
        # self.log("Example embedding:{}".format(self.int_to_words(self.inverse_map(word_to_int),valid_embedded)))
        
        # Build the LSTM LM with softmax
        # embedding_matrix = np.array(int_to_emb)
        # self.max_vocab, _ = embedding_matrix.shape
        # self.model = self.build_graph(embedding_matrix)

        # data generators
        # train_generator = KerasBatchGenerator(train_embedded, self.max_vocab, skip_step=5)
        # valid_data_generator = KerasBatchGenerator(valid_embedded, self.max_vocab, skip_step=5)

        # model_save_name = '/model-{epoch:02d}.hdf5'
        # if self.continue_training:
        #     self.log("Loading model from {}".format(self.model_path))
        #     self.model = load_model(self.model_path)
        #     model_save_name = '/model-cont-{epoch:02d}.hdf5'
        #
        # checkpointer = ModelCheckpoint(filepath=self.save_path + model_save_name, verbose=1)
        #
        # self.model.fit_generator(train_generator.generate(),
        #                      steps_per_epoch=train_generator.n_batches,#len(train_x) // (self.batch_size * self.max_seq_size),
        #                      epochs=self.num_epochs,
        #                      validation_data=valid_data_generator.generate(),
        #                      validation_steps=valid_data_generator.n_batches,
        #                      callbacks=[checkpointer]
        #                      )
        self.log("Loading file from={}".format(self.lang_model_model_path))
        self.model = load_model(self.lang_model_model_path)
        print(self.model.summary())

    def reduce_embedding(self, int_to_emb, word_to_int, word_to_emb, tokens):
        
        # All next to each other
        tokens = np.concatenate(tokens)
        
        # Find unique tokens and count them
        unsorted_uniques, unsorted_counts = np.unique(tokens, return_counts=True)
        
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
        
        return r_word_to_embed, r_word_to_int, r_int_to_emb
    
    def resolve(self, word, word_to_emb, emb_dim, verbose=False):
        self.total_tried += 1
        try:
            emb = word_to_emb[word]
        except:
            emb = np.random.rand(emb_dim)
            self.total_failed +=1
            if verbose:
                self.log("Was not able to find an embedding for:{}".format(word))
        return emb
        
    def clean_strings(self, cleaners, data):
        for cleaner in cleaners:
            data = np.vectorize(cleaner)(data)
        return data
        
    def tokenize_data(self, tokenizer, data, verbose=False):
        #data:      [n_stories]
        start = time.time()
        if verbose: 
            self.log("--Starting to tokenize--")
        res = np.array([tokenizer.tokenize(string) for string in data])
        if verbose:
            self.log("--Done tokenizing--{}\n".format(time.time()-start))
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

    def extract(self, data: np.ndarray):
        all_sentences = data[:,2:7]
        original_shape  = all_sentences.shape
        all_sentences = all_sentences.flatten()
        
        all_sentences = self.clean_strings([lambda x: x.lower(), lambda x: x.replace(".", " . ")], all_sentences)
        tokenized_sentences = self.tokenize_data(self.tokenizer, all_sentences, verbose=True)        
        embedded_sentences = np.array(embed_to_ints(tokenized_sentences, self.word_to_int, verbose=True))
        
        embedded_data = np.reshape(embedded_sentences, original_shape)
        embedded_endings  = embedded_data[:,-1]
        
        avg_probs_end = []
        
        #statistics
        pa = 1000
        start = time.time()
        tmr = time.time()
        done = 0
        #statistics
        
        for ending in embedded_endings:
            
            #get [n_batches, n_timesteps, n_vocab] --> [n_timesteps, n_vocab]
            prob = self.model.predict(np.array([ending]))[0]
            
            #resolve actual log probability
            avg = self.find_log_probability(prob, ending)
            
            #statistics
            done+=1
            if done % pa == 0:
                tm = time.time()
                tft = tm-tmr
                sps = done/(tm - start)
                remaining = len(embedded_endings)-done
                print("--------{}---------".format(tm-tmr ))
                print("Current speed:{}".format(pa/tft))
                print("Average pace:{}".format(done/(tm-start)))
                print("ETA:{} minutes".format(remaining/(60*sps)))
                print("Done:{}\n".format(done))
                tmr = time.time()
                print(self.int_to_words(self.inverse_map(self.word_to_int), [ending]))
            #statistics
            
            avg_probs_end.append(avg)
        
        #statistics
        pa = 1000
        start = time.time()
        tmr = time.time()
        done = 0
        #statistics
        
        avg_probs_full = []
        merged_full = self.merge_sentences_ints(embedded_data)
        
        for i, full_story in enumerate(merged_full):
            prob = self.model.predict(np.array([full_story]))[0]
            
            length_ending = len(embedded_endings[i])
            length_full = len(full_story)
            
            prob = prob[length_full - length_ending - 1:]
            
            avg = self.find_log_probability(prob, full_story)
            
            avg_probs_full.append(avg)
        
            #statistics
            done+=1
            if done % pa == 0:
                tm = time.time()
                tft = tm-tmr
                sps = done/(tm - start)
                remaining = len(merged_full)-done
                print("--------{}---------".format(tm-tmr ))
                print("Current speed:{}".format(pa/tft))
                print("Average pace:{}".format(done/(tm-start)))
                print("ETA:{} minutes".format(remaining/(60*sps)))
                print("Done:{}\n".format(done))
                tmr = time.time()
                print(self.int_to_words(self.inverse_map(self.word_to_int), [embedded_endings[i]]))
            #statistics
            
        conditional = [a - b for (a,b) in zip(avg_probs_full, avg_probs_end)]
        
        return np.column_stack((avg_probs_end, avg_probs_full, conditional))
        
    def find_log_probability(self, probabilities, ints):
        #probabilities: [n_timesteps, n_vocab]
        #ints:          [n_timesteps]       ints in range of vocab size
        
        log_prob = 0
        for time_step in range(len(probabilities)):
            word_distr = probabilities[time_step]
            word = ints[time_step]
            probability = word_distr[word]
            log_prob += math.log(probability)
        
        avg_log = log_prob / len(probabilities)
        
        return avg_log
        
    def merge_sentences_ints(self, data):
        return np.array([self.combine_lists(sample) for sample in data])
        
    def combine_lists(self, list_of_lists):
        return [item for sublist in list_of_lists for item in sublist]
        
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
                batches = self.create_batch(l_data, self.preferred_batch_size)
                for b in batches:
                    yield b[0], b[1]

    def create_batch(self, data, pref_batch_size):
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
