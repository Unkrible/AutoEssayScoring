from common import log, timeit
from models.feature_common import *
from ingestion.dataset import *
import re
from functools import reduce
import pandas as pd
import pickle

import language_check
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
from models.syntax import SyntaxFeature
from models.feature_engineering import FeatureEngineer

class SyntaxFeatureEngineer(FeatureEngineer):
    def __init__(self):
        FeatureEngineer.__init__(self)
        pass

    def fit(self, data):
        pass

    @timeit
    def transform(self, data):
        data=self._syntax_features(data)
        return data

    def fit_transform(self, data):
        self.fit(data)
        data=self.transform(data)
        #print(data['sents'])
        return data

    # syntax features:
    # measuring the average depths of the trees
    # and measure the proportion of causal and temporal clauses
    @timeit
    def _syntax_features(self, data):
        syn_feature=SyntaxFeature()
        '''
        #syntax_tree_depth_feature
        tree_depth_list=[]
        for essay in data['sents']:
            depth_list=[]
            for sentence in essay:
                depth=syn_feature.get_tree_depth(sentence)
                depth_list.append(depth)
            tree_depth_list.append(sum(depth_list) / len(depth_list))
        data['syntax_tree_depth']=tree_depth_list
        print(data['syntax_tree_depth'])
        '''
        #temporal_clauses_ratio_feature
        temporal_clauses_ratio_list=[]
        for essay in data['sents']:
            temporal_clauses_ratio=0
            for sentence in essay:
                if syn_feature.is_temporal_clauses(sentence):
                    temporal_clauses_ratio+=1
            #print(temporal_clauses_ratio,len(essay))
            temporal_clauses_ratio_list.append(temporal_clauses_ratio/len(essay))
        data['temporal_clauses_ratio']=temporal_clauses_ratio_list
        #print(data['temporal_clauses_ratio'])
        #print(data['sents'][0])

        # causal_clauses_ratio_feature
        causal_clauses_ratio_list = []
        for essay in data['sents']:
            causal_clauses_ratio = 0
            for sentence in essay:
                if syn_feature.is_causal_clauses(sentence):
                    causal_clauses_ratio += 1
            #print(causal_clauses_ratio , len(essay))
            causal_clauses_ratio_list.append(causal_clauses_ratio / len(essay))
        data['causal_clauses_ratio'] = causal_clauses_ratio_list
        #print(data['causal_clauses_ratio'])
        return data

FeatureDatasets = namedtuple("FeatureDatasets", ['train', 'train_label', 'valid', 'valid_label', 'test'])

if __name__ == "__main__":

    for set_id in range(1, 9):
        print("Now for set %d" % set_id)
        dataset=pickle.load(open('../resources/dataframes/TaskIndependentFeatureLabelSet'+str(set_id)+'.pl', 'rb'))

        train = dataset[0]
        train_label=dataset[1]
        valid=dataset[2]
        valid_label=dataset[3]
        test=dataset[4]

        fe = SyntaxFeatureEngineer()
        train = fe.fit_transform(train)
        #print(train['intj'])

        valid = fe.fit_transform(valid)
        test = fe.fit_transform(test)
        to_save = FeatureDatasets(train=train, train_label=train_label, valid=valid, valid_label=valid_label, test = test)
        pickle.dump(to_save, open('../resources/dataframes2/SyntaxFeatureLabelSet'+str(set_id)+'.pl', 'wb'))
        #break
    print("Done~")

