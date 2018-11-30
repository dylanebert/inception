import os
import pickle
import numpy as np
import h5py

predictions_dir = '/data/nlp/aux/predictions'
prediction_files = os.listdir(predictions_dir)

with h5py.File('/home/dylan/Documents/inception/model/gmc/predictions_compressed', 'w') as f:
    dset = f.create_dataset('predictions', (1500000, 18518), chunks=(1,18518), dtype=np.float16)
    idx = 0
    for i, filename in enumerate(prediction_files):
        print((i, idx), end='\r')
        with open(os.path.join(predictions_dir, filename), 'rb') as f:
            for filename, predictions in pickle.load(f).items():
                dset[idx,:] = predictions
                idx += 1
