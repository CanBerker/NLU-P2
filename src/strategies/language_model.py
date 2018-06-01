import keras as kr
import numpy as np
import string
import time

from keras.preprocessing.text import Tokenizer as tk
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
 
from strategies import Strategy


class LanguageModelStrategy(Strategy):

    def log(self, line):
        print("[LSTM] {0}".format(line))
        
    def fit(self, data: np.ndarray) -> None:
        
        self.max_vocab = 5000
        self.oov_token = "<unk>"
        self.embedding_size = 50
        self.hidden_size = 50
        self.use_dropout = True
        self.dropout_rate = 0.5
        self.batch_size = 32
        self.train_size = 0.8
        self.optimizer = Adam()
        self.num_epochs = 10
        self.save_path = "."
        
        _ = data[:,0]
        _ = data[:,1]
        partial = data[:,2:6]
        endings = data[:,6]
        
        full_stories = data[:,2:7]
        
        stories = self.merge_sentences(full_stories)
        self.fit_tokenizer(stories)
        
        word_to_int, int_to_word = self.get_int_mappings()
                
        embedded_stories = self.embed_data(stories, word_to_int)
        
        self.max_seq_sz = self.find_max_seq(embedded_stories)
        
        embedded_stories = self.pad_data(embedded_stories, self.max_seq_sz)
                
        model = self.build_graph()
        
        train_x, validation_x = train_test_split(embedded_stories, train_size = self.train_size)
        
        train_generator = KerasBatchGenerator(train_x,
                                        self.max_seq_sz,
                                        self.batch_size, 
                                        self.max_vocab,
                                        skip_step=5)
        valid_data_generator = KerasBatchGenerator(validation_x,
                                        self.max_seq_sz,
                                        self.batch_size, 
                                        self.max_vocab,
                                        skip_step=5)
                                                    
        checkpointer = ModelCheckpoint(filepath=self.save_path + '/model-{epoch:02d}.hdf5', verbose=1)
        
        model.fit_generator(train_generator.generate(),
                            len(train_x)//(self.batch_size*self.max_seq_sz), 
                            self.num_epochs,
                            validation_data=valid_data_generator.generate(),
                            validation_steps=len(validation_x)//(self.batch_size*self.max_seq_sz), 
                            callbacks=[checkpointer])                         
        

    def build_graph(self):
        self.log("Building graph")
        vocab_size = self.max_vocab
        model = Sequential()
        model.add(Embedding(vocab_size, self.embedding_size, input_length=self.max_seq_sz))
        model.add(LSTM(self.hidden_size, return_sequences=True))
        #model.add(LSTM(hidden_size, return_sequences=True))
        if self.use_dropout:
            model.add(Dropout(self.dropout_rate))
        model.add(TimeDistributed(Dense(vocab_size)))
        model.add(Activation('softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['categorical_accuracy'])
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
        return np.array(self.tokenizer.texts_to_sequences(text_data))
        
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
        m_len = -1
        for s in data:
            if len(s) > m_len:
                m_len = len(s)
                
        return m_len

class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step
    
    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                #Current index 
                x[i,:] = self.data[self.current_idx+i][0:-1]
                
            yield x, y