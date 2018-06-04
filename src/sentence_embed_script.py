from skip_thoughts import skipthoughts
import readers
import os
import csv
import numpy as np
import time
import math

train_data_loc = os.path.join(os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'), 'train.csv')
validation_data_loc = os.path.join(os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data'), 'validation.csv')

with open(train_data_loc, 'r') as f:
    csv_read = csv.reader(f)
    # Remove the csv header
    next(csv_read)
    train_data = np.array([np.array(line) for line in csv_read])

train_data = train_data[:,2:7]
train_data = np.array([' '.join(x) for x in train_data])

print(train_data)

samples = []
    

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
start = time.time()
print("--Starting to sentence embed the data")

bs =  32
all_embs = []
for i in range(int(math.ceil(len(train_data)/bs))):
    x_b = train_data[i*bs:(i+1)*bs]
    all_embs.extend(encoder.encode(train_data))
    print("Batch{}".format(i))
print("Took: {}".format(time.time() - start))

