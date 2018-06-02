import math
import math
import time

import numpy as np
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
        self.max_vocab = 1200000
        self.oov_token = "<unk>"
        self.embedding_size = 100
        self.hidden_size = 50
        self.use_dropout = False
        self.dropout_rate = 0.5
        # self.batch_size = 64
        self.train_size = 0.8
        self.optimizer = Adam()
        self.num_epochs = 200
        self.save_path = "/home/marenato/Documents/workspacePhd/NLU-P2/checkpoint/"
        glove_path = "/home/marenato/Documents/workspacePhd/NLU-P2/glove.twitter.27B.25d.txt"
        
        _ = data[:,0]
        _ = data[:,1]
        partial = data[:,2:6]
        endings = data[:,6]
        
        full_stories = data[:,2:7]

        word_to_emb, word_to_int, int_to_emb = load_glove(glove_path)

        stories_words = self.merge_sentences(full_stories)
        # print(stories_words.shape)
        tokenizer = nltk.tokenize.TreebankWordTokenizer()
        # tokenizer
        tokenized = np.array([tokenizer.tokenize(story) for story in stories_words])

        embedded_stories = embed_to_ints(tokenized, word_to_int=word_to_int)

        embedding_matrix = self.select_embeddings(tokenized, word_to_int)

        # embedded_stories = self.embed_data(stories_words, word_to_int)

                
        model = self.build_graph(embedding_matrix)
        
        train_x, validation_x = train_test_split(embedded_stories, train_size = self.train_size)
        
        train_generator = KerasBatchGenerator(train_x,
                                              self.max_vocab,
                                              skip_step=5)
        valid_data_generator = KerasBatchGenerator(validation_x,
                                                   self.max_vocab,
                                                   skip_step=5)

        checkpointer = ModelCheckpoint(filepath=self.save_path + '/model-{epoch:02d}.hdf5', verbose=1)

        # model.fit_generator(train_generator.generate(),
        #                     steps_per_epoch=train_generator.n_batches,#len(train_x) // (self.batch_size * self.max_seq_size),
        #                     epochs=self.num_epochs,
        #                     validation_data=valid_data_generator.generate(),
        #                     validation_steps=1,#len(validation_x)//(self.batch_size) ,#len(validation_x)//(self.batch_size * self.max_seq_size),
        #                     callbacks=[checkpointer]
        #                     )
        # self.test_model(train_x, int_to_word)

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
            prediction = model.predict(data[0])
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
        vocab_size = self.max_vocab
        model = Sequential()
        model.add(Embedding(vocab_size, self.embedding_size, embeddings_initializer=Constant(value=embedding_matrix)))
        model.add(LSTM(self.hidden_size, return_sequences=True))
        #model.add(LSTM(hidden_size, return_sequences=True))
        if self.use_dropout:
            model.add(Dropout(self.dropout_rate))
        model.add(TimeDistributed(Dense(vocab_size)))
        model.add(Activation('softmax'))
        
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['categorical_accuracy'])
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
        self.preferred_batch_size = 32
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

            lg, l_data = self.grouped_by_length[self.current_idx]
            batches = self.createBatch(l_data, self.preferred_batch_size)
            self.current_idx = (self.current_idx + 1) % len(self.grouped_by_length)
            for b in batches:
                yield b[0], b[1]


    def createBatch(self, data, pref_batch_size):
        data = np.array(data)
        num_batches = math.ceil(len(data)/pref_batch_size)
        batches = []
        for i in range(num_batches):
            data_slice = data[i*pref_batch_size: (i+1)*pref_batch_size]
            x_slice = data_slice[:,:-1]
            tmp_y = data_slice[:,1:]
            y_slice = to_categorical(tmp_y, num_classes=self.vocabulary)

            batches.append((x_slice, y_slice))

        return batches

class PredictCallback(Callback):
    def predict_on_batch_end(self, epoch, logs):
        data = ['kelly', 'found', 'her', "grandmother's", 'pizza', 'recipe', 'in', 'a', 'of', 'memories', 'kelly', 'about', 'how', 'much', 'she', 'loved', 'her', "grandmother's", 'pizza', 'kelly', 'decided', 'that', 'she', 'was', 'going', 'to', 'try', 'to', 'make', 'pizza', 'kelly', 'studied', 'the', 'recipe', 'and', 'gathered', 'everything', 'she', 'needed', 'kelly', 'successfully', 'made', 'a', 'pizza', 'from', 'her', "grandmother's", 'recipe']

        # self.model