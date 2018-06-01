from __future__ import print_function
import os
import sys
import numpy as np
# np.random.seed(1337)

import keras
from keras.models import Model, Sequential, load_model
from keras.regularizers import l1_l2, l2
from keras.layers import Embedding, Reshape, LSTM, Dense, concatenate, average, add, MaxPooling1D, TimeDistributed, Flatten, Lambda, Input, Dropout, Bidirectional, add, Masking
from keras.optimizers import Adam, RMSprop, SGD
import keras.backend as K
from src.ntm import SimpleNTM as NTM

import scipy.spatial.distance as distance
from scipy.cluster.hierarchy import linkage, inconsistent, fcluster
# from scipy.cluster.hierarchy import fclusterdata

EMBEDDING_DIM = 300
MAX_DISTANCE = 15 #40
MAXLEN = 10 + 10 # max length of entities allowed
BATCH_SIZE = 5

def get_pre_ntm_model(group_size=None, input_dropout=0.3, max_len=MAXLEN, embedding_matrix=None, pos_size=49, **kwargs):
    """group_size here is the # of pairs in a group"""

    # Shared embedding layer
    print("embedding matrix", embedding_matrix.shape)
    word_embedding = Embedding(len(embedding_matrix), EMBEDDING_DIM, mask_zero=False, weights=[embedding_matrix], trainable=True)
    pos_embedding = Embedding(pos_size+1, 10, mask_zero=False, trainable=True) # index 0 reserved for mask

    input_distance = Input(shape=(group_size, 1))
    input_speaker = Input(shape=(group_size, 1))

    input_word_current = Input(shape=(group_size, max_len))
    word_current_emb = word_embedding(input_word_current)
    word_current_emb = Dropout(input_dropout)(word_current_emb)
    word_current_lstm = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False, activation='linear'), merge_mode='ave'))(word_current_emb)

    input_pos_current = Input(shape=(group_size, max_len))
    pos_current_emb = pos_embedding(input_pos_current)
    pos_current_emb = Dropout(input_dropout)(pos_current_emb)
    pos_current_lstm = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False, activation='linear'), merge_mode='ave'))(pos_current_emb)

    input_word_prev = Input(shape=(group_size, max_len))
    word_prev_emb = word_embedding(input_word_prev)
    word_prev_emb = Dropout(input_dropout)(word_prev_emb)
    word_prev_lstm = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False, activation='linear'), merge_mode='ave'))(word_prev_emb)

    input_pos_prev = Input(shape=(group_size, max_len))
    pos_prev_emb = pos_embedding(input_pos_prev)
    pos_prev_emb = Dropout(input_dropout)(pos_prev_emb)
    pos_prev_lstm = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False, activation='linear'), merge_mode='ave'))(pos_prev_emb)

    concat = concatenate([input_distance, input_speaker, word_current_lstm, pos_current_lstm, word_prev_lstm, pos_prev_lstm])
    hidden1 = Dense(512, activation='relu')(concat)
    hidden1 = Dropout(0.3)(hidden1)
    hidden2 = Dense(256, activation='tanh')(hidden1)
    hidden2= Dropout(0.3)(hidden2)
    outlayer = Dense(1, activation='sigmoid', name='pairwise_out')(hidden2)

    model = Model(inputs=[input_distance, input_speaker, input_word_current, input_pos_current, input_word_prev, input_pos_prev], outputs=[outlayer])
    # model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

    return model


def get_pre_ntm_model2(group_size=None, input_dropout=0.3, max_len=MAXLEN, embedding_matrix=None, pos_size=49, **kwargs):
    """group_size here is the # of pairs in a group"""

    # Shared layers
    word_embedding = Embedding(len(embedding_matrix), EMBEDDING_DIM, mask_zero=False, weights=[embedding_matrix], trainable=True)
    pos_embedding = Embedding(pos_size+1, 10, mask_zero=False, trainable=True) # index 0 reserved for mask
    word_lstm = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False, activation='linear'), merge_mode='ave'))
    pos_lstm = TimeDistributed(Bidirectional(LSTM(16, return_sequences=False, activation='linear'), merge_mode='ave'))

    input_distance = Input(shape=(group_size, 1))
    input_speaker = Input(shape=(group_size, 1))

    input_word_current = Input(shape=(group_size, max_len))
    word_current_emb = word_embedding(input_word_current)
    word_current_emb = Dropout(input_dropout)(word_current_emb)
    # word_current_lstm = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False, activation='linear'), merge_mode='ave'))(word_current_emb)
    word_current_lstm = word_lstm(word_current_emb)

    input_pos_current = Input(shape=(group_size, max_len))
    pos_current_emb = pos_embedding(input_pos_current)
    pos_current_emb = Dropout(input_dropout)(pos_current_emb)
    # pos_current_lstm = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False, activation='linear'), merge_mode='ave'))(pos_current_emb)
    pos_current_lstm = pos_lstm(pos_current_emb)

    input_word_prev = Input(shape=(group_size, max_len))
    word_prev_emb = word_embedding(input_word_prev)
    word_prev_emb = Dropout(input_dropout)(word_prev_emb)
    # word_prev_lstm = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False, activation='linear'), merge_mode='ave'))(word_prev_emb)
    word_prev_lstm = word_lstm(word_prev_emb)

    input_pos_prev = Input(shape=(group_size, max_len))
    pos_prev_emb = pos_embedding(input_pos_prev)
    pos_prev_emb = Dropout(input_dropout)(pos_prev_emb)
    # pos_prev_lstm = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False, activation='linear'), merge_mode='ave'))(pos_prev_emb)
    pos_prev_lstm = pos_lstm(pos_prev_emb)

    concat = concatenate([input_distance, input_speaker, word_current_lstm, pos_current_lstm, word_prev_lstm, pos_prev_lstm])
    hidden1 = Dense(512, activation='relu')(concat)
    hidden1 = Dropout(0.3)(hidden1)
    hidden2 = Dense(256, activation='tanh')(hidden1)
    hidden2= Dropout(0.3)(hidden2)
    outlayer = Dense(1, activation='sigmoid', name='pairwise_out')(hidden2)

    model = Model(inputs=[input_distance, input_speaker, input_word_current, input_pos_current, input_word_prev, input_pos_prev], outputs=[outlayer])
    # model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0005), metrics=['accuracy'])

    return model


def get_combined_ntm_model(batch_size=5, group_size=40, input_dropout=0.3, max_len=MAXLEN, embedding_matrix=None, pos_size=49,
                           m_depth=256, n_slots=100, ntm_output_dim=128, shift_range=3, read_heads=1, write_heads=1, **kwargs):
    # Shared embedding layer
    print("embedding matrix", embedding_matrix.shape)
    word_embedding = Embedding(len(embedding_matrix), EMBEDDING_DIM, mask_zero=False, weights=[embedding_matrix], trainable=True)
    pos_embedding = Embedding(pos_size+1, pos_size+1, embeddings_initializer='identity', mask_zero=False, trainable=True) # index 0 reserved for mask

    input_distance = Input(shape=(group_size, 1))
    input_speaker = Input(shape=(group_size, 1))

    input_word_current = Input(shape=(group_size, max_len))
    word_current_emb = word_embedding(input_word_current)
    word_current_emb = Dropout(input_dropout)(word_current_emb)
    word_current_lstm = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False, activation='linear'), merge_mode='ave'))(word_current_emb)

    input_pos_current = Input(shape=(group_size, max_len))
    pos_current_emb = pos_embedding(input_pos_current)
    pos_current_emb = Dropout(input_dropout)(pos_current_emb)
    pos_current_lstm = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False, activation='linear'), merge_mode='ave'))(pos_current_emb)

    input_word_prev = Input(shape=(group_size, max_len))
    word_prev_emb = word_embedding(input_word_prev)
    word_prev_emb = Dropout(input_dropout)(word_prev_emb)
    word_prev_lstm = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False, activation='linear'), merge_mode='ave'))(word_prev_emb)

    input_pos_prev = Input(shape=(group_size, max_len))
    pos_prev_emb = pos_embedding(input_pos_prev)
    pos_prev_emb = Dropout(input_dropout)(pos_prev_emb)
    pos_prev_lstm = TimeDistributed(Bidirectional(LSTM(128, return_sequences=False, activation='linear'), merge_mode='ave'))(pos_prev_emb)

    concat = concatenate([input_distance, input_speaker, word_current_lstm, pos_current_lstm, word_prev_lstm, pos_prev_lstm])
    hidden1 = TimeDistributed(Dense(512, activation='relu'))(concat)

    encoder = concatenate([word_current_lstm, word_prev_lstm, hidden1])
    NTM_F = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_stateful=False, key_range=256,
                return_sequences=True, stateful=False, activation='relu', batch_size=batch_size)
    NTM_B = NTM(ntm_output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
                read_heads=read_heads, write_heads=write_heads, controller_stateful=False, key_range=256,
                return_sequences=True, stateful=False, activation='relu', go_backwards=True, batch_size=batch_size)

    ntm_forward = NTM_F(encoder)
    ntm_backward = NTM_B(encoder)

    ntm_layer = average([ntm_forward, ntm_backward])
    ntm_layer = Dropout(0.3)(ntm_layer)

    hidden_ntm = TimeDistributed(Dense(512, activation='tanh'))(ntm_layer)
    hidden_ntm = Dropout(0.3)(hidden_ntm)
    decoder = TimeDistributed(Dense(128, activation='sigmoid'))(hidden_ntm)
    decoder = Dropout(0.3)(decoder)

    outlayer = TimeDistributed(Dense(1, activation='sigmoid', name='pairwise_out'))(decoder)

    inputs = [input_distance, input_speaker, input_word_current, input_pos_current, input_word_prev, input_pos_prev]
    model = Model(inputs=inputs, outputs=[outlayer])
    # model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

    return model


def get_triad_model(group_size=None, input_dropout=0.3, max_len=MAXLEN, embedding_matrix=None, pos_size=49, **kwargs):
    """group_size here is the # of pairs in a group"""

    WordEmbedding = Embedding(len(embedding_matrix), EMBEDDING_DIM, mask_zero=False, weights=[embedding_matrix], trainable=True)
    PosEmbedding = Embedding(pos_size+1, pos_size+1, embeddings_initializer='identity', mask_zero=False, trainable=True) # index 0 reserved for mask
    # WordLSTM = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='ave'))
    # WordMaxPool = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))
    WordLSTM = Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='ave')
    WordMaxPool = MaxPooling1D(pool_size=max_len, padding='same')
    WordReduceDim = Lambda(lambda x: K.max(x, axis=-2))
    # PosLSTM = TimeDistributed(Bidirectional(LSTM(16, return_sequences=True, activation='linear'), merge_mode='ave'))
    # PosMaxPool = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))
    PosLSTM = Bidirectional(LSTM(16, return_sequences=True, activation='linear'), merge_mode='ave')
    PosMaxPool = MaxPooling1D(pool_size=max_len, padding='same')
    PosReduceDim = Lambda(lambda x: K.max(x, axis=-2))

    Hidden_1 = Dense(128, activation='relu')
    Hidden_2 = Dense(64, activation='relu')
    Decoder = Dense(32, activation='relu', name='decoder')

    # input_distance0 = Input(shape=(group_size, 1))
    # input_distance1 = Input(shape=(group_size, 1))
    # input_distance2 = Input(shape=(group_size, 1))
    input_distance0 = Input(shape=(1,))
    input_distance1 = Input(shape=(1,))
    input_distance2 = Input(shape=(1,))
    input_distances = [input_distance0, input_distance1, input_distance2]

    # input_speaker0 = Input(shape=(group_size, 1))
    # input_speaker1 = Input(shape=(group_size, 1))
    # input_speaker2 = Input(shape=(group_size, 1))
    input_speaker0 = Input(shape=(1,))
    input_speaker1 = Input(shape=(1,))
    input_speaker2 = Input(shape=(1,))
    input_speakers = [input_speaker0, input_speaker1, input_speaker2]

    input_word_0 = Input(shape=(max_len,))
    word_emb_0 = WordEmbedding(input_word_0)
    word_emb_0 = Dropout(input_dropout)(word_emb_0)
    word_lstm_0 = WordLSTM(word_emb_0)
    word_lstm_0 = WordMaxPool(word_lstm_0)
    word_lstm_0 = WordReduceDim(word_lstm_0)

    input_word_1 = Input(shape=(max_len,))
    word_emb_1 = WordEmbedding(input_word_1)
    word_emb_1 = Dropout(input_dropout)(word_emb_1)
    word_lstm_1 = WordLSTM(word_emb_1)
    word_lstm_1 = WordMaxPool(word_lstm_1)
    word_lstm_1 = WordReduceDim(word_lstm_1)

    input_word_2 = Input(shape=(max_len,))
    word_emb_2 = WordEmbedding(input_word_2)
    word_emb_2 = Dropout(input_dropout)(word_emb_2)
    word_lstm_2 = WordLSTM(word_emb_2)
    word_lstm_2 = WordMaxPool(word_lstm_2)
    word_lstm_2 = WordReduceDim(word_lstm_2)

    input_words = [input_word_0, input_word_1, input_word_2]

    input_pos_0 = Input(shape=(max_len,))
    pos_emb_0 = PosEmbedding(input_pos_0)
    pos_emb_0 = Dropout(input_dropout)(pos_emb_0)
    pos_lstm_0 = PosLSTM(pos_emb_0)
    pos_lstm_0 = PosMaxPool(pos_lstm_0)
    pos_lstm_0 = PosReduceDim(pos_lstm_0)

    input_pos_1 = Input(shape=(max_len,))
    pos_emb_1 = PosEmbedding(input_pos_1)
    pos_emb_1 = Dropout(input_dropout)(pos_emb_1)
    pos_lstm_1 = PosLSTM(pos_emb_1)
    pos_lstm_1 = PosMaxPool(pos_lstm_1)
    pos_lstm_1 = PosReduceDim(pos_lstm_1)

    input_pos_2 = Input(shape=(max_len,))
    pos_emb_2 = PosEmbedding(input_pos_2)
    pos_emb_2 = Dropout(input_dropout)(pos_emb_2)
    pos_lstm_2 = PosLSTM(pos_emb_2)
    pos_lstm_2 = PosMaxPool(pos_lstm_2)
    pos_lstm_2 = PosReduceDim(pos_lstm_2)

    input_pos_tags = [input_pos_0, input_pos_1, input_pos_2]

    concat01 = concatenate([input_distance0, input_speaker0, word_lstm_0, pos_lstm_0, word_lstm_1, pos_lstm_1])
    hidden01_1 = Hidden_1(concat01)
    hidden01_2 = Hidden_2(Dropout(0.3)(hidden01_1))

    concat12 = concatenate([input_distance1, input_speaker1, word_lstm_1, pos_lstm_1, word_lstm_2, pos_lstm_2])
    hidden12_1 = Hidden_1(concat12)
    hidden12_2 = Hidden_2(Dropout(0.3)(hidden12_1))

    concat20 = concatenate([input_distance2, input_speaker2, word_lstm_2, pos_lstm_2, word_lstm_0, pos_lstm_0])
    hidden20_1 = Hidden_1(concat20)
    hidden20_2 = Hidden_2(Dropout(0.3)(hidden20_1))

    hidden_shared = add([hidden01_2, hidden12_2, hidden20_2])
    hidden_3 = Dense(64, activation='relu', name='shared')(hidden_shared)

    decoder0 = Decoder(concatenate([hidden01_2, hidden_3]))
    # out0 = Dense(1, activation='sigmoid', name='out0')(Dropout(0.3)(decoder0))

    decoder1 = Decoder(concatenate([hidden12_2, hidden_3]))
    # out1 = Dense(1, activation='sigmoid', name='out1')(Dropout(0.3)(decoder1))

    decoder2 = Decoder(concatenate([hidden20_2, hidden_3]))
    # out2 = Dense(1, activation='sigmoid', name='out2')(Dropout(0.3)(decoder2))

    decoder_all = concatenate([decoder0, decoder1, decoder2])
    outlayer = Dense(3, activation='sigmoid', name='out')(Dropout(0.3)(decoder_all))

    model = Model(inputs=input_distances+input_speakers+input_words+input_pos_tags, outputs=[outlayer])
    # model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])

    return model


def clustering(pair_results, binarize=False):
    def distance(e1, e2):
        e1 = tuple(e1.astype(int))
        e2 = tuple(e2.astype(int))
        if e1 == e2:
            return 1.0  # This is the minumum distance
        if (e1, e2) in pair_results:
            similarity = max(pair_results[(e1, e2)], 1e-3)
            # dist = 1 - pair_results[(e1, e2)] #+ 1e-4
            dist = min(1.0 / (similarity), 10.0)
            # dist = (10 * (1 - pair_results[(e1, e2)])) ** 2
        else:
            # dist = 0.9
            dist = 10.0
        if binarize:
            dist = np.round(dist)

        return dist

    # distance has no direction
    if sys.version_info[0] == 3:
        pairs = list(pair_results.keys())
    else:
        pairs = pair_results.keys()
    for key in pairs:
        pair_results[(key[1], key[0])] = pair_results[key]

    x = [key[0] for key in pair_results]
    x = list(set(x))
    x.sort(key=lambda x: x[0])
    x = np.array(x)

    clusters, Z = fclusterdata(x, 1.7, criterion='distance', metric=distance, depth=2, method='single')
    return x, clusters, Z


def fclusterdata(X, t, criterion='inconsistent',
                     metric='euclidean', depth=2, method='single', R=None):
    """
    This is adapted from scipy fclusterdata.
    https://github.com/scipy/scipy/blob/v1.0.0/scipy/cluster/hierarchy.py#L1809-L1878
    """
    X = np.asarray(X, order='c', dtype=np.double)

    if type(X) != np.ndarray or len(X.shape) != 2:
        print(type(X), X.shape)
        raise TypeError('The observation matrix X must be an n by m numpy '
                        'array.')

    Y = distance.pdist(X, metric=metric)
    Z = linkage(Y, method=method)
    if R is None:
        R = inconsistent(Z, d=depth)
    else:
        R = np.asarray(R, order='c')
    T = fcluster(Z, criterion=criterion, depth=depth, R=R, t=t)
    return T, Z