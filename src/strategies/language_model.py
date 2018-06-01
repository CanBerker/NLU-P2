import keras as kr
import numpy as np
import time

from keras.preprocessing.text import Tokenizer as tk
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
 
from strategies import Strategy


class LanguageModelStrategy(Strategy):

    @staticmethod
    def log(line):
        print("[LSTM] {0}".format(line))
        
    def fit(self, data: np.ndarray) -> None:
        
        self.max_vocab = 5000
        self.oov_token = "<unk>"
        self.embedding_size = 50
        self.hidden_size = 50
        self.use_dropout = True
        self.dropout_rate = 0.5
        self.batch_size = 64
        self.train_size = 0.8
        self.optimizer = Adam()
        self.num_epochs = 10
        self.save_path = "."
        
        _ = data[:,0]
        _ = data[:,1]
        partial = data[:,2:6]
        endings = data[:,6]
        
        full_stories = data[:,2:7]
        
        stories_words = self.merge_sentences(full_stories)
        self.fit_tokenizer(stories_words)
        
        word_to_int, int_to_word = self.get_int_mappings()
                
        embedded_stories = self.embed_data(stories_words, word_to_int)
        self.max_seq_size = 30
        #self.max_seq_size, self.min_seq_size = self.find_max_seq(embedded_stories)
        
        #embedded_stories = self.pad_data(embedded_stories, self.max_seq_size)
                
        model = self.build_graph()
        
        train_x, validation_x = train_test_split(embedded_stories, train_size = self.train_size)
        
        train_generator = KerasBatchGenerator(train_x,
                                              self.max_seq_size,
                                              self.batch_size,
                                              self.max_vocab,
                                              skip_step=5)
        valid_data_generator = KerasBatchGenerator(validation_x,
                                                   self.max_seq_size,
                                                   self.batch_size,
                                                   self.max_vocab,
                                                   skip_step=5)
                                                    
        checkpointer = ModelCheckpoint(filepath=self.save_path + '/model-{epoch:02d}.hdf5', verbose=1)
        
        model.fit_generator(train_generator.generate(),
                            len(train_x) // (self.batch_size * self.max_seq_size),
                            self.num_epochs,
                            validation_data=valid_data_generator.generate(),
                            validation_steps=len(validation_x)//(self.batch_size * self.max_seq_size),
                            callbacks=[checkpointer])
        self.test_model(train_x, int_to_word)

    def test_model(self, train_data, reversed_dictionary):
        model = load_model(self.save_path + "/model-10.hdf5")
        dummy_iters = 40
        example_training_generator = KerasBatchGenerator(train_data, self.max_seq_size, 1, self.max_vocab, skip_step=1)
        print("Training data:")
        for i in range(dummy_iters):
            dummy = next(example_training_generator.generate())
        num_predict = 10
        true_print_out = "Actual words: "
        pred_print_out = "Predicted words: "
        for i in range(num_predict):
            data = next(example_training_generator.generate())
            prediction = model.predict(data[0])
            predict_word = np.argmax(prediction[:, self.max_seq_size-1, :])
            true_print_out += reversed_dictionary[train_data[self.max_seq_size + dummy_iters + i]] + " "
            pred_print_out += reversed_dictionary[predict_word] + " "
        print(true_print_out)
        print(pred_print_out)

    def build_graph(self):
        self.log("Building graph")
        vocab_size = self.max_vocab
        model = Sequential()
        model.add(Embedding(vocab_size, self.embedding_size, input_length=self.max_seq_size))
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
        tmp = np.apply_along_axis(lambda x: ' '.join(x), 1, data)
        list_of_words = []
        for s in tmp:
            list_of_words.extend(s.split(" "))
        return list_of_words
        
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
        return [word_to_int[w] for w in text_data if w in word_to_int ]
        #return self.tokenizer.texts_to_sequences(text_data)
        
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
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
            x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
            temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
            # convert all of temp_y into a one hot representation
            y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
            self.current_idx += self.skip_step
            yield x, y