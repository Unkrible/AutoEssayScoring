
def _clean_en_text(text_with_index, re_space, stopwords):
    text, index = text_with_index
    text = re_space.sub(' ', text).lower()

    filtered_words = text.split(' ')
    filtered_words = list(map(lambda word: word.strip(), filtered_words))
    if stopwords:
        filtered_words = [w for w in filtered_words if w.strip() not in stopwords]
    return ' '.join(filtered_words), index
