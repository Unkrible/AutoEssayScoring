from common import log, timeit
from models.feature_common import *
from ingestion.dataset import *
import re
from functools import reduce
import pandas as pd
import pickle
import json

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
        data = self._syntax_features(data)
        return data

    def fit_transform(self, data):
        self.fit(data)
        data = self.transform(data)
        # print(data['sents'])
        return data

    # syntax features:
    # measuring the average depths of the trees
    # and measure the proportion of causal and temporal clauses
    @timeit
    def _syntax_features(self, data):
        syn_feature = SyntaxFeature()
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
        # temporal_clauses_ratio_feature
        temporal_clauses_ratio_list = []
        for essay in data['sents']:
            temporal_clauses_ratio = 0
            for sentence in essay:
                if syn_feature.is_temporal_clauses(sentence):
                    temporal_clauses_ratio += 1
            # print(temporal_clauses_ratio,len(essay))
            temporal_clauses_ratio_list.append(temporal_clauses_ratio / len(essay))
        data['temporal_clauses_ratio'] = temporal_clauses_ratio_list
        # print(data['temporal_clauses_ratio'])
        # print(data['sents'][0])

        # causal_clauses_ratio_feature
        causal_clauses_ratio_list = []
        for essay in data['sents']:
            causal_clauses_ratio = 0
            for sentence in essay:
                if syn_feature.is_causal_clauses(sentence):
                    causal_clauses_ratio += 1
            # print(causal_clauses_ratio , len(essay))
            causal_clauses_ratio_list.append(causal_clauses_ratio / len(essay))
        data['causal_clauses_ratio'] = causal_clauses_ratio_list
        # print(data['causal_clauses_ratio'])
        return data


def normalization(df, ignore=None, eps=0.00000001):
    columns_name = df.columns.values.tolist()
    for each in columns_name:
        MIN, MAX = min(df[each]), max(df[each])
        if each in ignore:
            continue
        d = df[each].apply(lambda x: (x-MIN)/(MAX-MIN+eps))
        df.drop(each, axis=1)
        df[each] = d
    return df


def read_test_labels(test_label_path, set_index):
    df = pd.read_csv(test_label_path, sep='\t')
    for each in df.columns.values.tolist():
        if each not in ['essay_id', 'essay_set', 'domain1_score', 'rater1_domain1', 'rater2_domain1']:
            df.pop(each)
    df = df[df['essay_set'] == set_index]
    df.pop('essay_set')
    return df


FeatureDatasets = namedtuple("FeatureDatasets", ['train', 'train_label', 'valid', 'valid_label', 'test'])

if __name__ == "__main__":

    for set_id in range(1, 9):
        print("Now for set %d" % set_id)
        dataset = pickle.load(
            open('../resources/dataframes/TaskIndependentFeatureLabelSet' + str(set_id) + '.pl', 'rb'))

        train = dataset.train
        valid = dataset.valid
        test = dataset.test

        fe = SyntaxFeatureEngineer()
        train = fe.fit_transform(train)
        valid = fe.fit_transform(valid)
        test = fe.fit_transform(test)

        train_label = read_test_labels('../resources/essay_data/train.tsv', set_id)
        valid_label = read_test_labels('../resources/essay_data/dev.tsv', set_id)
        test_label = read_test_labels('../resources/essay_data/test.tsv', set_id)

        for each in ['tokens', 'sents', 'lemma', 'pos', 'ner', 'matches', 'corrected', 'essay']:
            train.pop(each)
        for each in ['tokens', 'sents', 'lemma', 'pos', 'ner', 'matches', 'corrected', 'essay']:
            valid.pop(each)
        for each in ['tokens', 'sents', 'lemma', 'pos', 'ner', 'matches', 'corrected', 'essay']:
            test.pop(each)

        train_len, valid_len, test_len = len(train), len(valid), len(test)
        all_data = pd.concat([train, valid, test])
        all_data = normalization(all_data, ignore=['essay_id'])
        train, valid, test = \
            all_data[:train_len], all_data[train_len: train_len+valid_len], all_data[train_len+valid_len:]

        assert(len(train)==len(train_label) and len(valid)==len(valid_label) and len(test)==test_len)

        # to_save = FeatureDatasets(train=train, train_label=train_label, valid=valid, valid_label=valid_label, test=test)
        train.to_csv('../resources/dataframes2/TrainSet' + str(set_id) + '.csv', index=False)
        train_label.to_csv('../resources/dataframes2/TrainLabel' + str(set_id) + '.csv', index=False)

        valid.to_csv('../resources/dataframes2/ValidSet' + str(set_id) + '.csv', index=False)
        valid_label.to_csv('../resources/dataframes2/ValidLabel' + str(set_id) + '.csv', index=False)

        test.to_csv('../resources/dataframes2/TestSet' + str(set_id) + '.csv', index=False)
        test_label.to_csv('../resources/dataframes2/TestLabel'+str(set_id)+'.csv', index=False)
        # pickle.dump(to_save, open('../resources/dataframes2/SyntaxFeatureLabelSet' + str(set_id) + '.pl', 'wb'))
        # break
    print("Done~")
