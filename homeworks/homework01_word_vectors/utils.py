import numpy as np
from matplotlib import pyplot as plt
from collections import Counter, deque


def plot_train_process(train_loss, val_loss, train_accuracy, val_accuracy, title_suffix=''):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].set_title(' '.join(['Loss', title_suffix]))
    axes[0].plot(train_loss, label='train')
    axes[0].plot(val_loss, label='validation')
    axes[0].legend()

    axes[1].set_title(' '.join(['Validation accuracy', title_suffix]))
    axes[1].plot(train_accuracy, label='train')
    axes[1].plot(val_accuracy, label='validation')
    axes[1].legend()
    plt.show()


class Splitter:
    def __init__(self, ngrams=(1,)):
        self.ngrams = ngrams

    @staticmethod
    def get_ngrams(text, n=1):
        window = deque(maxlen=n)
        ngram_list = list()
        for word in text.split():
            window.append(word)
            if len(window) == n:
                ngram_list.append(' '.join(window))
        ngram_counter = Counter(ngram_list)
        return ngram_counter

    def split_text(self, text):
        ngram_list = Counter()
        for n in self.ngrams:
            ngram_list.update(self.get_ngrams(text, n))
        return ngram_list


class BagOfWords:
    def __init__(self, dataset, max_feat=None, ngrams=(1,)):
        self.splitter = Splitter(ngrams)
        self.word_counter = Counter()
        for text in dataset:
            self.word_counter.update(self.splitter.split_text(text))
        self.vocabulary = None
        self.set_max_feat(max_feat)

    def set_max_feat(self, max_feat=None):
        self.vocabulary = self.word_counter.most_common(max_feat)
        self.vocabulary = [word for word, _ in self.vocabulary]

    def transform(self, text):
        text_counter = Counter(text.split())
        freq_feature = [text_counter[word] for word in self.vocabulary]
        return np.array(freq_feature, 'float32')


class TFiDF:
    def __init__(self, dataset, alpha, max_feat=None, ngrams=(1,)):
        self.splitter = Splitter(ngrams)
        self.dataset_length = len(dataset)
        self.alpha = alpha
        self.df_counter = self.compute_df(dataset)
        self.vocabulary = None
        self.idf = None
        self.set_max_feat(max_feat)

    def transform(self, text):
        text_counter = Counter(self.splitter.split_text(text))
        tf = [text_counter[item] for item in self.vocabulary]
        tf = np.array(tf)
        tfidf = tf * self.idf
        return tfidf.astype('float32')

    def compute_df(self, dataset):
        split_dataset = np.vectorize(lambda x: self.splitter.split_text(x))(dataset)
        vocabulary = set()
        for text_ngram_collection in split_dataset:
            vocabulary.update(text_ngram_collection.keys())
        df = {item: np.vectorize(lambda t: item in t)(split_dataset).sum()
              for item in vocabulary}
        return Counter(df)

    def set_max_feat(self, max_feat=None):
        top_df_item_list = self.df_counter.most_common(max_feat)
        self.vocabulary = [item for item, _ in top_df_item_list]
        self.idf = np.array([freq for _, freq in top_df_item_list]) + self.alpha
        self.idf = np.log(self.dataset_length) - np.log(self.idf)


class TextVectorizer:
    def __init__(self, word_vectorizer, tokenizer, text_maxlen):
        self.word_vectorizer = word_vectorizer
        self.tokenizer = tokenizer
        self.text_maxlen = text_maxlen

    def vectorize_text(self, text):
        vector = []
        for word in self.tokenizer.tokenize(text):
            if word in self.word_vectorizer.key_to_index:
                vector.append(self.word_vectorizer[word])
        if len(vector) < self.text_maxlen:
            pad_size = self.text_maxlen - len(vector)
            vector.extend([np.zeros(self.word_vectorizer.vector_size) for _ in range(pad_size)])
        elif len(vector) > self.text_maxlen:
            vector = vector[:self.text_maxlen]
        vector = np.array(vector)
        return vector

    def transform(self, dataset):
        text_tensor = np.vectorize(self.vectorize_text, otypes=[np.ndarray])(dataset)
        text_tensor = np.array(text_tensor.tolist())
        return text_tensor


