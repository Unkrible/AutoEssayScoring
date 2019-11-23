import pickle
from collections import namedtuple

FeatureDatasets = namedtuple("FeatureDatasets", ['train', 'train_label', 'valid', 'valid_label', 'test'])

for set_id in range(1, 9):
    print("Now for set %d" % set_id)
    dataset = pickle.load(open('../resources/dataframes2/SyntaxFeatureLabelSet' + str(set_id) + '.pl', 'rb'))

    train = dataset[0]
    print(train['sents'])
    print(train['intj'])
    print(train['causal_clauses_ratio'])
    break
