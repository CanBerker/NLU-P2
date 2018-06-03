import numpy as np
import os.path


# data: data to augment -> train data
# ratio: negative samples to be created per positive sample
def augment_data(X, ratio = 4, load=False):
    fname = "augmented_data_%d.npy" % ratio
    if (os.path.isfile(fname)) and load:
        augmented_data = np.load(fname)
    else:
        beginnings = X[:,0:6]
        endings = X[:,6]
        
        # create copies of the rows
        tile_parameter = (ratio, 1)
        augmented_data = np.tile(beginnings, tile_parameter)
        random_endings = endings[np.random.choice(len(endings), size=len(augmented_data), replace=True)]
        augmented_data = np.column_stack((augmented_data, random_endings))
        
        # save as npy binary
        np.save(fname, augmented_data)
        
    return augmented_data



