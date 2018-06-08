from skip_thoughts import skipthoughts
import readers
import os
import csv
import numpy as np
import time
import math

#File must be in /data
DATA_FILE = "validation.csv"
DATA_FORMAT = "VALIDATION"
OUTPUT_NAME = "valid_embedding"


#all = np.load(OUTPUT_NAME + ".npy")
#all = all[:,3:]
#np.save(OUTPUT_NAME + "_last", all)

#quit()

data_loc = os.path.join(os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'), DATA_FILE)

with open(data_loc, 'r') as f:
    csv_read = csv.reader(f)
    # Remove the csv header
    next(csv_read)
    data = np.array([np.array(line) for line in csv_read])

if DATA_FORMAT == "TRAIN":
    data = data[:,5:7]
elif DATA_FORMAT == "VALIDATION":
    # 1-4: Partial_story
    # 5  : ending 1
    # 6  : ending 2
    data = data[:,4:7]
    
n_samples, n_sentences = data.shape

print("Sentence data:{}".format(data.shape))
data = data.flatten()
print("Flattened:{}".format(data.shape))


model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

start = time.time()
print("--Starting to sentence embed the data")
sentences_embedded = encoder.encode(data)
print("Took: {}".format(time.time() - start))

res = np.reshape(sentences_embedded, (n_samples, n_sentences, -1))

print("Embedded data shape:{}".format(np.array(sentences_embedded).shape))

print("Starting to save the data...")
np.save(OUTPUT_NAME, res)
print("Done saving!")

