from abc import ABC, abstractmethod
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
    """Class for splitting text into ngrams."""
    def __init__(self, ngrams=(1,)):
        self.ngrams = ngrams

    @staticmethod
    def get_ngrams(text, n=1):
        """Split text into sequences of length n

        :param text: text to split
        :param n: length of sequences
        :return: Counter
        """
        window = deque(maxlen=n)
        ngram_list = list()
        for word in text.split():
            window.append(word)
            if len(window) == n:
                ngram_list.append(' '.join(window))
        ngram_counter = Counter(ngram_list)
        return ngram_counter

    def split_text(self, text):
        """Split text into sequences of length in self.ngrams

        :param text: text to split
        :return: Counter
        """
        ngram_list = Counter()
        for n in self.ngrams:
            ngram_list.update(self.get_ngrams(text, n))
        return ngram_list


class TextTransformer(ABC):
    """TextTransformer interface"""
    @abstractmethod
    def transform_text(self, text):
        raise NotImplementedError

    def transform_dataset(self, dataset):
        return np.stack(list(map(self.transform_text, dataset)))


class BagOfWords(TextTransformer):
    """Class to transform text into vector of words frequencies"""
    def __init__(self, dataset, max_feat=None, ngrams=(1,)):
        self.splitter = Splitter(ngrams)
        self.word_counter = Counter()
        for text in dataset:
            self.word_counter.update(self.splitter.split_text(text))
        self.vocabulary = None
        self.set_max_feat(max_feat)

    def transform_text(self, text):
        text_counter = Counter(text.split())
        freq_feature = [text_counter[word] for word in self.vocabulary]
        return np.array(freq_feature, 'float32')

    def set_max_feat(self, max_feat=None):
        self.vocabulary = self.word_counter.most_common(max_feat)
        self.vocabulary = [word for word, _ in self.vocabulary]


class TFiDF(TextTransformer):
    """Class to transform text via TFiDF"""
    def __init__(self, dataset, alpha, max_feat=None, ngrams=(1,)):
        self.splitter = Splitter(ngrams)
        self.dataset_length = len(dataset)
        self.alpha = alpha
        self.df_counter = self.compute_df(dataset)
        self.vocabulary = None
        self.idf = None
        self.set_max_feat(max_feat)

    def transform_text(self, text):
        text_counter = Counter(self.splitter.split_text(text))
        tf = [text_counter[item] for item in self.vocabulary]
        tf = np.array(tf)
        tfidf = tf * self.idf
        return tfidf.astype('float32')

    def set_max_feat(self, max_feat=None):
        top_df_item_list = self.df_counter.most_common(max_feat)
        self.vocabulary = [item for item, _ in top_df_item_list]
        self.idf = np.array([freq for _, freq in top_df_item_list]) + self.alpha
        self.idf = np.log(self.dataset_length) - np.log(self.idf)

    def compute_df(self, dataset):
        split_dataset = np.vectorize(lambda x: self.splitter.split_text(x))(dataset)
        vocabulary = set()
        for text_ngram_collection in split_dataset:
            vocabulary.update(text_ngram_collection.keys())
        df = {item: np.vectorize(lambda t: item in t)(split_dataset).sum()
              for item in vocabulary}
        return Counter(df)


class Word2Vec(TextTransformer):
    """Class to transform text with use of word2vec model"""
    def __init__(self, word2vec_model, tokenizer, text_maxlen):
        self.word2vec_model = word2vec_model
        self.tokenizer = tokenizer
        self.text_maxlen = text_maxlen

    def transform_text(self, text):
        text_vector = []
        for word in self.tokenizer.tokenize(text):
            if word in self.word2vec_model.key_to_index:
                word_vector = self.word2vec_model[word]
            else:
                word_vector = np.zeros(self.word2vec_model.vector_size)
            text_vector.append(word_vector)
        if len(text_vector) < self.text_maxlen:
            pad_size = self.text_maxlen - len(text_vector)
            text_vector.extend([np.zeros(self.word2vec_model.vector_size) for _ in range(pad_size)])
        elif len(text_vector) > self.text_maxlen:
            text_vector = text_vector[:self.text_maxlen]
        text_vector = np.stack(text_vector, axis=1)
        return text_vector

    def set_text_maxlen(self, text_maxlen):
        self.text_maxlen = text_maxlen
