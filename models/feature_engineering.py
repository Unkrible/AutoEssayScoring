from common import log, timeit
import re
from functools import reduce
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import math


def _clean_en_text(text_with_index, re_space, stopwords):
    text, index = text_with_index
    text = re_space.sub(' ', text).lower()

    filtered_words = text.split(' ')
    filtered_words = list(map(lambda word: word.strip(), filtered_words))
    if stopwords:
        filtered_words = [w for w in filtered_words if w.strip() not in stopwords]
    return ' '.join(filtered_words), index


def _word_freq(essay_words):
    word_freq = {}
    for each in essay_words:
        if each in word_freq:
            word_freq[each] += 1
        else:
            word_freq[each] = 1
    return word_freq


def _gen_tfidf_matrix(data, vector_length, stopwords):
    re_space = re.compile(r'[\s+\?\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+')
    clean_data = dict()
    # 去停词，去标点符号
    for i, text in data.items():
        text, i = _clean_en_text((text, i), re_space, stopwords)
        clean_data[i] = text
        
    # word_freq统计所有单词的词频, word_docs统计每个单词在多少文档中出现过
    word_freq, word_docs = dict(), dict()
    for (i, text) in clean_data.items():
        text_words = list(filter(lambda x: x!='', text.split(' ')))
        essay_word_freq = _word_freq(text_words)
        for item in essay_word_freq.items():
            if item[0] in word_freq:
                word_freq[item[0]] += item[1]
            else:
                word_freq[item[0]] = item[1]
            if item[0] in word_docs:
                word_docs[item[0]] += 1
            else:
                word_docs[item[0]] = 1
    word_freq_sorted = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    words_select = [each[0] for each in word_freq_sorted[:vector_length]]
    doc_freq = {k: word_docs[k] for k in words_select}

    tfidf = dict()
    for (i, text) in clean_data.items():
        text_words = list(filter(lambda x: x != '', text.split(' ')))
        vector = np.zeros(vector_length)
        essay_word_freq = _word_freq(text_words)
        for index in range(len(words_select)):
            word = words_select[index]
            if word in essay_word_freq:
                vector[index] = essay_word_freq[word]*math.log(len(data)/doc_freq[word], 10)
        tfidf[i] = vector
    return Series(tfidf)


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
        pass

    def fit_transform(self, data):
        self.fit(data)
        self.transform(data)

    # return average sentence length in words and word length in characters
    def _length_feature(self, essay):
        sentences = re.split(r'\.|\?|!', essay)
        sentences = [each.strip().split(' ') for each in sentences]
        sentence_length = [len(x) for x in sentences]
        word_length = [[len(x) for x in each] for each in sentences]
        word_length = reduce(lambda x, y: x+y, word_length)
        word_length = list(filter(lambda x: x != 0, word_length))
        return sum(sentence_length)/len(sentence_length), sum(word_length)/len(word_length)

    # the occurance of commas, quotations and exclamation marks
    def _occurrence_feature(self, essay):
        pass

    # syntax features:
    # measuring the ratio of distinct parse trees to
    # all the trees and the average depths of the trees
    def _syntax_features(self, essay):
        pass

    # the relative ratio of POS-tags to detect stylepreferences of writers
    def _style_features(self, essay):
        pass

    def _cohesion_features(self, essay):
        pass

    def _coherence_features(self,essay):
        pass

    def _error_features(self, essay):
        pass

    def _readability_features(self, essay):
        pass

    def _task_similarity_features(self, essay):
        pass

    def _set_dependent_features(self, essay):
        pass


if __name__ == "__main__":
    s = "I am a student. I love playing basketball! Do you love that?"
    #fe = TaskIndependentFeatureEngineer()
