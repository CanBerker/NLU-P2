import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier as rf

from extractors import Extractor

from utils.loader import join_and_shuffle
from utils.loader import get_labels

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed, Input
from keras.layers import LSTM
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer as tk
from keras.utils import to_categorical

class SentenceEmbeddingExtractor(Extractor):

    def __init__(self, embedded_train_path, embedded_val_path):
        self.embedded_train_path = embedded_train_path
        self.embedded_val_path = embedded_val_path
        self.n_sentences_keep = 2
        self.optimizer = Adam()
        
    # Expects the training set, not augmented.
    def fit(self, data: np.ndarray) -> None:
        #Fit wants the augmented set but then embedded to sentences
        
        #Decompose the data
        _     = data[:,0]
        _     = data[:,1]
        train = data[:,2:7]
        
        #Find the pre-embedded data
        embedded_train = np.load(self.embedded_train_path)        
        
        #Perform some checking for sanity
        self.perform_value_checks(train, embedded_train)
        
        #In order to avoid running an hour long script on predef-augmented set 
        #We augment the set here instead.
        embedded_aug, all_labels = self.augment_data(embedded_train)
        
        embedded_aug = np.array(embedded_aug)
        
        print("Augmented the training data to:{}".format(np.array(embedded_aug).shape))
        
        features = self.extract_features(embedded_aug)
        n_samples, n_features = features.shape
        
        self.model = self.build_model(n_features, 64)
        self.model.fit(features,
                       all_labels,
                       batch_size = 64,
                       epochs = 100,
                       validation_split = 0.2)
        
        pass

        
    # All extractors expect the validation set
    def extract(self, data: np.ndarray) -> str:
        #Decompose data
        #--> data[:,0] contains ID'sa
        #--> data[:,1] contains title
        #--> data[:,2-6] contains first 4 sentences
        #--> data[:,6] contains ending
        #--> data[:,7] contains labels
        
        return None

    def build_model(self, input_dim, batch_size):
        inputs = Input(shape=(input_dim,))
        #hl_1 = Dense(2400, activation='relu')(inputs)
        hl_2 = Dense(1200, activation='relu')(inputs)
        hl_2 = Dropout(0.5)(hl_2)
        hl_3 = Dense(300 , activation='relu')(hl_2)        
        hl_3 = Dropout(0.5)(hl_3)
        outputs = Dense(2, activation='softmax')(hl_3)
        
        model = Model(inputs, outputs)
        
        model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer, metrics=['sparse_categorical_accuracy'])
        print(model.summary())
        
        return model
        
    def augment_data(self, data, ratio=1):
        beginnings  = data[:,:4]
        endings     = data[:,4]
                
        # create copies of the rows
        tile_parameter = (ratio, 1)
        negative_samples = np.tile(beginnings, tile_parameter)
        random_endings = endings[np.random.choice(len(endings), size=len(negative_samples), replace=True)]
        random_endings = random_endings[:,np.newaxis,:] #newaxis to be able to stack
        negative_samples = np.column_stack((negative_samples, random_endings))
        
        negative_samples = negative_samples
        positive_samples = data
        
        negative_lables = get_labels(negative_samples, 0)
        positive_labels = get_labels(positive_samples, 1)
    
        all_stories, all_labels = join_and_shuffle([negative_samples, positive_samples],
                                            [negative_lables, positive_labels])
    
                
        return all_stories, all_labels
    
    def extract_features(self, data):
        # Here we decide what to take e.g. entire context, last sentence or no context.
        
        n_sentences_keep = self.n_sentences_keep
        n_samples, n_sentences, _ = data.shape
        cut_data = data[:,n_sentences - n_sentences_keep:]
        
        print("Cutting away {} sentences".format(n_sentences - n_sentences_keep))
        print("Keeping {} sentences".format(self.n_sentences_keep))
        
        #print(cut_data.shape)
        #print(cut_data[0])
        
        #Sum the data in sentence dimention
        print("Reducing by summing sentence embedding...")
        features = np.sum(cut_data, axis=1)        
        print("Features now have shape:{}".format(features.shape))
        #print(features[0])
        
        return features
        
        
    def perform_value_checks(self, train, embedded_train):
        n_samples, n_sentences = train.shape
        n_samples_e, n_sentences_e, emb_size = embedded_train.shape
        
        if n_sentences != n_sentences_e:
            raise ValueError("The embedded file doesn't contain equal amount of sentences per sample as the given data...")
            
        if n_samples_e != n_samples:
            raise ValueError("The embedded file doesn't contain equal amount of samples as the given data...")
        
        
        print("Embedded shape:{}".format(embedded_train.shape))
        print("Non Embedded shape:{}".format(train.shape))
        