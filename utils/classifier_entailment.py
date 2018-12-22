import h5py
import pickle
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input entailment pairs to evaluate', type=str, required=True)
parser.add_argument('-o', '--output', help='output results file', type=str, required=True)
parser.add_argument('-k', '--top_k', help='filter to top K predictions', type=int, default=5)
args = parser.parse_args()

wordnet_pairs = []
with open(args.input, 'r') as f:
    for line in f:
        w1, w2 = line.rstrip().split(' ')
        wordnet_pairs.append((w1, w2))

with open('/data/nlp/aux/index_class_dict.p', 'rb') as f:
    index_class_dict = pickle.load(f)
    class_index_dict = {v: k for k, v in index_class_dict.items()}

entails = {}
for pair in wordnet_pairs:
    entails[pair] = [0, 0]

with h5py.File('/home/dylan/Documents/inception/model/gmc/predictions', 'r') as f:
    predictions = f['predictions']
    for i in tqdm(range(len(predictions))):
        p = predictions[i]
        top_idx = np.argsort(p)[::-1][:args.top_k]
        for pair in wordnet_pairs:
            w1, w2 = pair
            if class_index_dict[w1] in top_idx and class_index_dict[w2] in top_idx:
                if p[class_index_dict[w2]] > p[class_index_dict[w1]]:
                    entails[pair][0] += 1
                else:
                    entails[pair][1] += 1

with open(args.output, 'w+') as f:
    for pair, vals in entails.items():
        w1, w2 = pair
        entails, entails_not = vals
        f.write('{0}\t{1}\t{2}\t{3}\n'.format(w1, w2, entails, entails_not))
