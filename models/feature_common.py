import re
from collections import Counter
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer
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
    return dict(Counter(essay_words))


def gen_tfidf_matrix(data, vector_length, stopwords):
    re_space = re.compile(r'[\s+\?\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+')
    clean_data = dict()
    # 去停词，去标点符号
    for i, text in data.items():
        text, i = _clean_en_text((text, i), re_space, stopwords)
        clean_data[i] = text

    # word_freq统计所有单词的词频, word_docs统计每个单词在多少文档中出现过
    word_freq, word_docs = dict(), dict()
    for (i, text) in clean_data.items():
        text_words = list(filter(lambda x: x != '', text.split(' ')))
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
                vector[index] = essay_word_freq[word] * math.log(len(data) / doc_freq[word], 10)
        tfidf[i] = vector
    return Series(tfidf)


# 对essay做词性标注，返回词和词性元组的list
def pos_tag(words_list):
    # english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    # words_list = [word for word in essay_words if word not in english_punctuations]
    return [each[1] for each in nltk.pos_tag(words_list)]


# the relative ratio of POS-tags to detect style preferences of writers
# together with the type token ratio to detect the diversity of vocabulary
def style_features(x):
    pos_tag_list = x['pos']
    A, B = 0, 0
    for each in pos_tag_list:
        if each in ['NOUN', 'ADJ', 'DET']:
            A += 1
        if each in ['PRON', 'VERB', 'ADV']:
            B += 1
    N = x['token_count']
    formal_feature = (A / N - B / N + 100) / 2
    return formal_feature
