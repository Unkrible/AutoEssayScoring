from common import log, timeit
import re
from functools import reduce
import pandas as pd
from pandas import Series, DataFrame


def _clean_en_text(text_with_index, re_space, stopwords):
    text, index = text_with_index
    text = re_space.sub(' ', text).lower()

    filtered_words = text.split(' ')
    filtered_words = list(map(lambda word: word.strip(), filtered_words))
    if stopwords:
        filtered_words = [w for w in filtered_words if w.strip() not in stopwords]
    return ' '.join(filtered_words), index

# def _gen_tfidf_matrix(data, vector_length, stopwords):
#     re_space = re.compile(r'[\s+\?\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+')
#


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
    fe = TaskIndependentFeatureEngineer()

