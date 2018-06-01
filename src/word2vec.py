from __future__ import print_function
import numpy as np


def load_glove(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

# The folliwing functions are adapted from https://github.com/facebookresearch/InferSent/blob/master/data.py

def get_glove(word_dict, glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}(/{1}) words with glove vectors'.format(
                len(word_vec), len(word_dict)))
    return word_vec

def build_vocab_from_tokens(tokens, glove_path):
    """Build vocab from a list of tokens"""
    word_dict = {}
    for tok in tokens:
        if tok not in word_dict:
            word_dict[tok] = ''
    word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec

def get_glove_k(K, glove_path):
    """create word_vec with k first glove vectors.
    Assuming word vectors are sorted by frequency. glove.840B.300d.txt satisfies
    """
    k = 0
    word_vec = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if k <= K:
                word_vec[word] = np.fromstring(vec, sep=' ')
                k += 1
            if k > K:
                if word in ['<s>', '</s>']:
                    word_vec[word] = np.fromstring(vec, sep=' ')

            if k > K and all([w in word_vec for w in ['<s>', '</s>']]):
                break
    return word_vec


def build_vocab(tokens, glove_path, K=1000):
    """build a vocabulary to include K most frequent tokens as long as those in passed-in tokens"""
    word_vec = get_glove_k(K, glove_path)
    word_dict = {}
    for tok in tokens:
        if tok not in word_vec and tok not in word_dict:
            word_dict[tok] = ''
    word_vec.update(get_glove(word_dict, glove_path))
    print("vocab size: %d " % len(word_vec))

    return word_vec
