from common import log, timeit
from models.feature_common import *
from ingestion.dataset import *
import re
from functools import reduce
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer

import language_check
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
from string import punctuation


# TODO: 分词、去除停用词？、序列化、提取句法特征、提取词法特征(tf-idf)、提取语义特征
class FeatureEngineer:
    def __init__(self):
        pass

    @timeit
    def fit(self, data):
        pass

    @timeit
    def transform(self, data):
        pass

    def fit_transform(self, data):
        self.fit(data)
        self.transform(data)


"""
Features proposed in paper:
Task-Independent Features for Automated Essay Grading
Visit https://www.researchgate.net/publication/278383803_Task-Independent_Features_for_Automated_Essay_Grading
for details. 
"""
class TaskIndependentFeatureEngineer(FeatureEngineer):
    def __init__(self):
        FeatureEngineer.__init__(self)
        pass

    @timeit
    def fit(self, data):
        pass

    @timeit
    def transform(self, data):
        data = data[:10]
        data = pd.DataFrame({'essay_id': data.index, 'essay': data.values})

        #纠正词法句法错误
        tool = language_check.LanguageTool('en-US')
        data['matches'] = data['essay'].apply(lambda v: tool.check(v))
        data['corrections_num'] = data.apply(lambda l: len(l['matches']), axis=1)
        data['corrected'] = data.apply(lambda l: language_check.correct(l['essay'], l['matches']), axis=1)

        # 分词，对其做词性标注，命名实体识别
        tokens, sents, lemma, pos, ner, stop_words = [], [], [], [], [], STOP_WORDS
        nlp = spacy.load('en_core_web_sm')
        for essay in nlp.pipe(data['corrected'], batch_size=2, n_threads=2):
            if essay.is_parsed:
                tokens.append([e.text for e in essay])
                sents.append([sent.string.strip() for sent in essay.sents])
                pos.append([e.pos_ for e in essay])
                ner.append([e.text for e in essay.ents])
                lemma.append([n.lemma_ for n in essay])
            else:
                tokens.append(None)
                sents.append(None)
                pos.append(None)
                ner.append(None)
                lemma.append(None)
        data['tokens'], data['sents'], data['lemma'], data['pos'], data['ner'] = tokens, sents, lemma, pos, ner

        # 提取各种特征
        data['token_count'] = data.apply(lambda x: len(x['tokens']), axis=1)
        data['unique_token_count'] = data.apply(lambda x: len(set(x['tokens'])), axis=1)
        data['type_token_ratio'] = data.apply(lambda x: x['unique_token_count']/x['token_count'], axis=1)
        data['sent_count'] = data.apply(lambda x: len(x['sents']), axis=1)
        data['ner_count'] = data.apply(lambda x: len(x['ner']), axis=1)
        data['comma'] = data.apply(lambda x: x['corrected'].count(','), axis=1)
        data['quotation'] = data.apply(lambda x: x['corrected'].count('\'') + x['corrected'].count('\"'), axis=1)
        data['exclamation'] = data.apply(lambda x: x['corrected'].count('!'), axis=1)

        data['organization'] = data.apply(lambda x: x['corrected'].count(r'@ORGANIZATION'), axis=1)
        data['caps'] = data.apply(lambda x: x['corrected'].count(r'@CAPS'), axis=1)
        data['person'] = data.apply(lambda x: x['corrected'].count(r'@PERSON'), axis=1)
        data['location'] = data.apply(lambda x: x['corrected'].count(r'@LOCATION'), axis=1)
        data['money'] = data.apply(lambda x: x['corrected'].count(r'@MONEY'), axis=1)
        data['time'] = data.apply(lambda x: x['corrected'].count(r'@TIME'), axis=1)
        data['date'] = data.apply(lambda x: x['corrected'].count(r'@DATE'), axis=1)
        data['percent'] = data.apply(lambda x: x['corrected'].count(r'@PERCENT'), axis=1)
        data['noun'] = data.apply(lambda x: x['pos'].count('NOUN'), axis=1)
        data['adj'] = data.apply(lambda x: x['pos'].count('ADJ'), axis=1)
        data['pron'] = data.apply(lambda x: x['pos'].count('PRON'), axis=1)
        data['verb'] = data.apply(lambda x: x['pos'].count('VERB'), axis=1)
        data['noun'] = data.apply(lambda x: x['pos'].count('NOUN'), axis=1)
        data['cconj'] = data.apply(lambda x: x['pos'].count('CCONJ'), axis=1)
        data['adv'] = data.apply(lambda x: x['pos'].count('ADV'), axis=1)
        data['det'] = data.apply(lambda x: x['pos'].count('DET'), axis=1)
        data['propn'] = data.apply(lambda x: x['pos'].count('PROPN'), axis=1)
        data['num'] = data.apply(lambda x: x['pos'].count('NUM'), axis=1)
        data['part'] = data.apply(lambda x: x['pos'].count('PART'), axis=1)
        data['intj'] = data.apply(lambda x: x['pos'].count('INTJ'), axis=1)

        data['formal_feature'] = data.apply(style_features, axis=1)
        
    def fit_transform(self, data):
        self.fit(data)
        self.transform(data)


    # syntax features:
    # measuring the ratio of distinct parse trees to
    # all the trees and the average depths of the trees
    # and measure the proportion of subordinate,
    # causal and temporal clauses
    def _syntax_features(self, essay):

        pass


if __name__ == "__main__":
    s = "I am a student. I love playing basketball! Do you love that?"

    dataset = AutoEssayScoringDataset("../resources/essay_data", 2)
    train, label = dataset.train
    fe = TaskIndependentFeatureEngineer()
    fe.transform(train)
    print("Done~")

