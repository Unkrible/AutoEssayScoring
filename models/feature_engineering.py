from common import log, timeit
from models.feature_common import _pos_tag, _gen_tfidf_matrix, _type_token_ratio
import re
from functools import reduce
import pandas as pd
import nltk


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
        return essay.count(','), essay.count('!'), essay.count("\'")+essay.count("\"")

    # syntax features:
    # measuring the ratio of distinct parse trees to
    # all the trees and the average depths of the trees
    # and measure the proportion of subordinate,
    # causal and temporal clauses
    def _syntax_features(self, essay):
        pass

    # the relative ratio of POS-tags to detect style preferences of writers
    # together with the type token ratio to detect the diversity of vocabulary
    def _style_features(self, essay):
        pos_tag_list = _pos_tag(essay)
        A, B = 0, 0
        for each in pos_tag_list:
            if each in []:
                A += 1
            if each in []:
                B += 1
        N = len(nltk.word_tokenize(essay))
        formal_feature = (A/N-B/N+100)/2
        return formal_feature, _type_token_ratio(essay)


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
    print(_pos_tag(s))
