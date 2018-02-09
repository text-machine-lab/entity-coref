from __future__ import print_function
import os
import sys
import numpy as np
# np.random.seed(1337)
import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.regularizers import l1_l2, l2
from keras.layers import Embedding, Reshape, LSTM, Dense, concatenate, average, add, MaxPooling1D, TimeDistributed, Flatten, Lambda, Input, Dropout, Bidirectional
from keras.optimizers import Adam, RMSprop, SGD
import keras.backend as K

EMBEDDING_DIM = 300
MAXLEN = 15 # max length of entities allowed

def get_pre_ntm_model(group_size=None, input_dropout=0.3, max_len=MAXLEN, embedding_matrix=None, pos_size=44, **kwargs):
    """group_size here is the # of pairs in a group"""

    # Shared embedding layer
    print("embedding matrix", embedding_matrix.shape)
    word_embedding = Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)
    pos_embedding = Embedding(pos_size, 10, trainable=True)

    input_distance = Input(shape=(group_size, 1))

    input_word_current = Input(shape=(group_size, max_len))
    word_current_emb = word_embedding(input_word_current)
    word_current_emb = Dropout(input_dropout)(word_current_emb)
    word_current_lstm = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(word_current_emb)
    word_current_lstm = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(word_current_lstm)
    word_current_lstm = Lambda(lambda x: K.max(x, axis=-2), name='word_current')(word_current_lstm)

    input_pos_current = Input(shape=(group_size, max_len))
    pos_current_emb = pos_embedding(input_pos_current)
    pos_current_emb = Dropout(input_dropout)(pos_current_emb)
    pos_current_lstm = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(pos_current_emb)
    pos_current_lstm = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(pos_current_lstm)
    pos_current_lstm = Lambda(lambda x: K.max(x, axis=-2), name='pos_current')(pos_current_lstm)

    input_word_prev = Input(shape=(group_size, max_len))
    word_prev_emb = word_embedding(input_word_prev)
    word_prev_emb = Dropout(input_dropout)(word_prev_emb)
    word_prev_lstm = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(word_prev_emb)
    word_prev_lstm = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(word_prev_lstm)
    word_prev_lstm = Lambda(lambda x: K.max(x, axis=-2), name='word_prev')(word_prev_lstm)

    input_pos_prev = Input(shape=(group_size, max_len))
    pos_prev_emb = pos_embedding(input_pos_prev)
    pos_prev_emb = Dropout(input_dropout)(pos_prev_emb)
    pos_prev_lstm = TimeDistributed(Bidirectional(LSTM(128, return_sequences=True, activation='linear'), merge_mode='sum'))(pos_prev_emb)
    pos_prev_lstm = TimeDistributed(MaxPooling1D(pool_size=max_len, padding='same'))(pos_prev_lstm)
    pos_prev_lstm = Lambda(lambda x: K.max(x, axis=-2), name='pos_prev')(pos_prev_lstm)

    concat = concatenate([input_distance, word_current_lstm, pos_current_lstm, word_prev_lstm, pos_prev_lstm])
    hidden1 = Dense(512, activation='relu')(concat)
    hidden1 = Dropout(0.3)(hidden1)
    hidden2 = Dense(256, activation='tanh')(hidden1)
    hidden2= Dropout(0.3)(hidden2)
    outlayer = Dense(1, activation='sigmoid', name='pairwise_out')(hidden2)

    model = Model(inputs=[input_distance, input_word_current, input_pos_current, input_word_prev, input_pos_prev], outputs=[outlayer])
    # model.summary()
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model