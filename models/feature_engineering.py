from common import log, timeit
from models.feature_common import *
from ingestion.dataset import *
import pickle
import re
from functools import reduce
import pandas as pd
import pickle

import language_check
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
from spacy_readability import Readability
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

    @timeit
    def fit(self, data):
        pass

    @timeit
    def transform(self, data):
        data = data[:]
        data = pd.DataFrame({'essay_id': data.index, 'essay': data.values})

        # 纠正词法句法错误
        tool = language_check.LanguageTool('en-US')
        data['matches'] = data['essay'].apply(lambda v: tool.check(v))
        data['corrections_num'] = data.apply(lambda l: len(l['matches']), axis=1)
        data['corrected'] = data.apply(lambda l: language_check.correct(l['essay'], l['matches']), axis=1)

        # 分词，对其做词性标注，命名实体识别
        tokens, sents, lemma, pos, ner, stop_words = [], [], [], [], [], STOP_WORDS
        flesch_kincaid_grade_level, flesch_kincaid_reading_ease, \
        dale_chall, smog, coleman_liau_index, automated_readability_index, \
        forcast = [], [], [], [], [], [], []
        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe(Readability())
        for essay in nlp.pipe(data['corrected'], batch_size=2, n_threads=2):
            if essay.is_parsed:
                tokens.append([e.text for e in essay])
                sents.append([sent.string.strip() for sent in essay.sents])
                pos.append([e.pos_ for e in essay])
                ner.append([e.text for e in essay.ents])
                lemma.append([n.lemma_ for n in essay])
                flesch_kincaid_grade_level.append(essay._.flesch_kincaid_grade_level)
                flesch_kincaid_reading_ease.append(essay._.flesch_kincaid_reading_ease)
                dale_chall.append(essay._.dale_chall)
                smog.append(essay._.smog)
                coleman_liau_index.append(essay._.coleman_liau_index)
                automated_readability_index.append(essay._.automated_readability_index)
                forcast.append(essay._.forcast)
            else:
                tokens.append(None)
                sents.append(None)
                pos.append(None)
                ner.append(None)
                lemma.append(None)
                flesch_kincaid_grade_level.append(None)
                flesch_kincaid_reading_ease.append(None)
                dale_chall.append(None)
                smog.append(None)
                coleman_liau_index.append(None)
                automated_readability_index.append(None)
                forcast.append(None)
        # 词性标注，命名实体识别，词根化，分词断句
        data['tokens'], data['sents'], data['lemma'], data['pos'], data['ner'] = tokens, sents, lemma, pos, ner
        # 可读性特征
        data['flesch_kincaid_grade_level'], data['flesch_kincaid_reading_ease'], data['dale_chall'], data['smog'], data[
            'coleman_liau_index'], data['automated_readability_index'], data['forcast'] = \
            flesch_kincaid_grade_level, flesch_kincaid_reading_ease, dale_chall, smog, coleman_liau_index, automated_readability_index, forcast

        # 提取各种特征
        data['token_count'] = data.apply(lambda x: len(x['tokens']), axis=1)
        data['unique_token_count'] = data.apply(lambda x: len(set(x['tokens'])), axis=1)
        data['type_token_ratio'] = data.apply(lambda x: x['unique_token_count'] / x['token_count'], axis=1)
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
        data['at_num'] = data.apply(lambda x: x['corrected'].count(r'@NUM'), axis=1)

        data['noun'] = data.apply(lambda x: x['pos'].count('NOUN'), axis=1)
        data['adj'] = data.apply(lambda x: x['pos'].count('ADJ'), axis=1)
        data['pron'] = data.apply(lambda x: x['pos'].count('PRON'), axis=1)
        data['verb'] = data.apply(lambda x: x['pos'].count('VERB'), axis=1)
        data['noun'] = data.apply(lambda x: x['pos'].count('NOUN'), axis=1)
        data['cconj'] = data.apply(lambda x: x['pos'].count('CCONJ'), axis=1)
        data['sconj'] = data.apply(lambda x: x['pos'].count('SCONJ'), axis=1)
        data['adv'] = data.apply(lambda x: x['pos'].count('ADV'), axis=1)
        data['det'] = data.apply(lambda x: x['pos'].count('DET'), axis=1)
        data['propn'] = data.apply(lambda x: x['pos'].count('PROPN'), axis=1)
        data['num'] = data.apply(lambda x: x['pos'].count('NUM'), axis=1)
        data['part'] = data.apply(lambda x: x['pos'].count('PART'), axis=1)
        data['intj'] = data.apply(lambda x: x['pos'].count('INTJ'), axis=1)
        data['aux'] = data.apply(lambda x: x['pos'].count('AUX'), axis=1)
        data['adp'] = data.apply(lambda x: x['pos'].count('ADP'), axis=1)
        data['punct'] = data.apply(lambda x: x['pos'].count('PUNCT'), axis=1)

        data['formal'] = data.apply(style_features, axis=1)

        connective_words = self._read_connective_words()
        data['cohesion'] = data.apply(lambda x: sum([1 if t in connective_words else 0 for t in x['tokens']]), axis=1)

        return data

    def fit_transform(self, data):
        self.fit(data)
        data = self.transform(data)
        return data

    def _read_connective_words(self):
        f = open("../resources/connective_words", 'r')
        connective_words = {word[:-1] for word in f.readlines()}
        return connective_words


FeatureDatasets = namedtuple("FeatureDatasets", ['train', 'train_label', 'valid', 'valid_label', 'test'])

if __name__ == "__main__":
    # s = "I am a student. I love playing basketball! Do you love that?"
    for set_id in range(1, 9):
        print("Now for set %d" % set_id)
        dataset = AutoEssayScoringDataset("../resources/essay_data", essay_set_id=set_id)
        train, train_label = dataset.train
        valid, valid_label = dataset.valid
        test = dataset.test.data

        fe = TaskIndependentFeatureEngineer()
        train = fe.fit_transform(train)
        valid = fe.fit_transform(valid)
        test = fe.fit_transform(test)
        to_save = FeatureDatasets(train=train, train_label=train_label, valid=valid, valid_label=valid_label, test=test)
        pickle.dump(to_save, open('../resources/dataframes/TaskIndependentFeatureLabelSet' + str(set_id) + '.pl', 'wb'))
    print("Done~")
