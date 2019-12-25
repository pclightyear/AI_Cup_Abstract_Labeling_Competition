import nltk
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def get_count_vect(docs, max_features=None, ngram_range=(1,1)):
    count_vect = CountVectorizer(max_features=max_features, ngram_range=ngram_range, lowercase=True)
    counts = count_vect.fit_transform(docs)

    return count_vect, counts

def get_tfidf_vect(docs, max_features=None, ngram_range=(1,1)):
    tfidf_vect = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, lowercase=True)
    tfidf = tfidf_vect.fit_transform(docs)

    return tfidf_vect, tfidf

def get_term_frequencies(matrix):
    term_freq = []

    for t in matrix.T:
        term_freq.append(t.toarray().sum())

    return term_freq

def plot_term_frequencies_sorted(seq, rank=-1, is_log=False):
    if rank is -1:
        rank = len(seq)
    index = np.arange(rank)

    if is_log:
        seq_sorted = np.sort([math.log(i) for i in seq])[::-1]
    else:
        seq_sorted = np.sort(seq)[::-1]

    plt.plot(index, seq_sorted[:rank])
    plt.show()

def plot_sparse_matrix(matrix, precision):
    plt.subplots(figsize=(20, 25))
    plt.spy(matrix, precision=precision, markersize=1)

def term_rank(freq, feature_names, rank, ascending=False):
    if ascending:
        return np.argsort(freq)[:rank]
    else:
        return np.argsort(freq)[::-1][:rank]

def format_rows(docs):
    """ format the text field and strip special characters """
    D = []
    for d in docs.data:
        temp_d = " ".join(d.split("\n")).strip('\n\t')
        D.append([temp_d])
    return D

def format_labels(target, docs):
    """ format the labels """
    return docs.target_names[target]

def check_missing_values(row):
    """ functions that check and verifies if there are missing values in dataframe """
    counter = 0
    for element in row:
        if element == True:
            counter+=1
    return ("The amoung of missing records is: ", counter)

def tokenize_text(text, remove_stopwords=False):
    """
    Tokenize text using the nltk library
    """
    tokens = []
    for d in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(d, language='english'):
            # filters here
            tokens.append(word)
    return tokens

stop_word_to_remove = [
'the',
'of',
'and',
'a',
'to',
'in',
'we',
'is',
'for',
'that',
'on',
'with',
'are',
'as',
'by',
'an',
'can',
'our',
'from',
'be',
'which',
'it',
'such',
'or',
'have',
'these',
'also',
'at',
'their',
'new',
'between',
'more',
'its',
'one',
'both',
'been',
'each',
'over',
'however',
'other',
'but',
'than',
'into',
'when',
'only',
'while',
"'s",
'where',
'how',
'most',
'all',
]

term_convert_dict_complex = {
"rnns": "recurrent neural network",
"dnns": "deep neural network",
"mnist": "handwritten digits database",
"autoencoder": "auto encoder",
"cifar-10": "image dataset",
"auto-encoder": "auto encoder",
"autoencoders": "auto encoders",
"sequence-to-sequence": "sequence to sequence",
"r-cnn": "region with convolutional neural network",
"resnets": "residual neural network",
"mclnn": "masked conditional neural network",
"image-to-image": "image to image",
"lte-u": "lte unlicensed spectrum",
"rgb-d": "rgb depth",
"fronthaul": "cloud radio access network",
"word2vec": "word to vector embedding model",
"dcnn": "deep convolutional neural network",
"sum-rate": "sum rate",
"-based": "based",
"cryptocurrencies": "crypto currencies",
"blstm": "bidirectional long short term memory",
"deep-learning": "deep learning",
"actor-critic": "reinforcement learning neural network",
"alexnet": "convolutional neural network",
"blocklength": "block length",
}

term_convert_dict_easy = {
"rnns": "neural network",
"dnns": "neural network",
"mnist": "dataset",
"autoencoder": "auto encoder",
"cifar-10": "dataset",
"auto-encoder": "auto encoder",
"autoencoders": "auto encoder",
"sequence-to-sequence": "sequence to sequence",
"r-cnn": "neural network",
"resnets": "neural network",
"mclnn": "neural network",
"image-to-image": "image to image",
"lte-u": "lte unlicensed spectrum",
"rgb-d": "rgb depth",
"fronthaul": "cloud radio access network",
"word2vec": "word embedding",
"dcnn": "neural network",
"sum-rate": "sum rate",
"-based": "based",
"cryptocurrencies": "crypto currencies",
"blstm": "neural network",
"deep-learning": "deep learning",
"actor-critic": "neural network",
"alexnet": "neural network",
"blocklength": "block length",
}