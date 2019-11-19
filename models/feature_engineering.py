from common import log, timeit


def _clean_en_text(text_with_index, re_space, stopwords):
    text, index = text_with_index
    text = re_space.sub(' ', text).lower()

    filtered_words = text.split(' ')
    filtered_words = list(map(lambda word: word.strip(), filtered_words))
    if stopwords:
        filtered_words = [w for w in filtered_words if w.strip() not in stopwords]
    return ' '.join(filtered_words), index


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
