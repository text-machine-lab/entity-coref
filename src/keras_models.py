"""keras on tensorflow"""
from __future__ import print_function
import os
import sys
import numpy as np
import json
import time
# np.random.seed(1337)

import keras
from keras.models import Model, Sequential, load_model
from keras.regularizers import l1_l2, l2
from keras.layers import Embedding, Reshape, LSTM, Dense, concatenate, average, add, MaxPooling1D, TimeDistributed, Flatten, Lambda, Input, Dropout, Bidirectional, add, Masking
from keras.optimizers import Adam, RMSprop, SGD
import keras.backend as K
from src.ntm import SimpleNTM as NTM
from src.build_data import build_dataFrame, DataGen, EMBEDDING_DIM, MAXLEN
from src.evaluator import TriadEvaluator


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

    concat20 = concatenate([input_distance2, input_speaker2, word_lstm_0, pos_lstm_0, word_lstm_2, pos_lstm_2])
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


def train(**kwargs):
    train_gen = kwargs['train_gen']
    val_dir = kwargs['val_dir']
    model_destination = kwargs['model_destination']
    epochs = kwargs['epochs']
    load_model = kwargs['load_model']


    group_size = 40
    train_input_gen = train_gen.generate_triad_input(file_batch=50)

    if val_dir is not None:
        # Need the same word indexes and pos indexes for training and test data
        val_gen = DataGen(build_dataFrame(val_dir, threads=1), train_gen.word_indexes, train_gen.pos_tags)
        val_input_gen = val_gen.generate_triad_input(file_batch=10, looping=True, threads=1)
        # just get data from 1 for try
        val_data_q = next(val_input_gen)
        val_data = val_data_q[0]
        # val_X, val_y = next(group_data(val_data, group_size, batch_size=None))
        val_X, val_y = val_data
        validation_split = 0
    else:
        validation_split = 0.2

    if load_model:
        model = load_model(os.path.join(model_destination, 'model.h5'))
    else:
        model = get_triad_model(group_size=group_size, input_dropout=0.5, max_len=MAXLEN,
                                embedding_matrix=train_gen.embedding_matrix, pos_size=len(train_gen.pos_tags))

    training_history = []
    evaluator = None

    # subproc_queue = multiprocessing.Queue(maxsize=100)
    # subproc = multiprocessing.Process(target=fill_data_q, args=(train_input_gen, subproc_queue))
    # subproc.daemon = True
    # subproc.start()

    for epoch in range(epochs):
        # train_data_q = subproc_queue.get()
        train_data_q = next(train_input_gen)
        n_training_files = len(train_data_q)
        epoch_history = []
        start = time.time()
        for n, data in enumerate(train_data_q):
            # for X, y in group_data(data, group_size, batch_size=None):
            X, y = data
            if not y.any(): continue

            history = model.fit(X, y, batch_size=100, epochs=1, verbose=0, validation_split=validation_split,
                                validation_data=[val_X, val_y], shuffle=False, class_weight=None, sample_weight=None,
                                initial_epoch=0)  # , callbacks=[tensorboard])
            if history.history:
                epoch_history.append(history.history)

            acc = np.mean([h['acc'] for h in epoch_history])
            loss = np.mean([h['loss'] for h in epoch_history])
            val_acc = np.mean([h['val_acc'] for h in epoch_history])
            val_loss = np.mean([h['val_loss'] for h in epoch_history])

            sys.stdout.write(
                "epoch %d after training file %d/%d--- -%ds - loss : %.4f - acc : %.4f - val_loss : %.4f - val_acc : %.4f\r" % (
                    epoch + 1, n + 1, n_training_files, int(time.time() - start), loss, acc, val_loss, val_acc))
            # sys.stdout.write("epoch %d after training file %d/%d--- -%ds - loss : %.4f - acc : %.4f\r" % (
            #     epoch + 1, n + 1, n_training_files, int(time.time() - start), loss, acc))
            sys.stdout.flush()

            if epoch > 99:
                K.set_value(model.optimizer.lr, 0.0005)
            if epoch > 149:
                K.set_value(model.optimizer.lr, 0.0001)

        # if epoch > 2 and args.val_dir is not None:
        #     evaluator.model = model
        #     val_acc = evaluator.fast_eval()
        #     sys.stdout.write("epoch %d after training file %d/%d--- -%ds - loss : %.4f - acc : %.4f - val_acc : %.4f\r" % (
        #         epoch + 1, n + 1, n_training_files, int(time.time() - start), loss, acc, val_acc))
        sys.stdout.write("\n")
        training_history.append({'categorical_accuracy': acc, 'loss': loss})
        if epoch > 0 and epoch % 10 == 0:
            try:
                model.save(os.path.join(model_destination, 'model.h5'))
            except:
                model.save_weights(os.path.join(model_destination, 'weights.h5'))

            if val_dir:
                if evaluator is None:
                    evaluator = TriadEvaluator(model, val_input_gen)
                    # evaluator.data_available = True
                    # filler = multiprocessing.Process(target=evaluator.fill_q_store, args=())
                    # filler.daemon = True
                    # filler.start()
                else:
                    evaluator.model = model

                eval_results = evaluator.fast_eval()
                print(eval_results)

    eval_results = evaluator.fast_eval()
    print(eval_results)
    # with open(os.path.join(args.model_destination, 'results.pkl'), 'w') as f:
    #     pickle.dump(eval_results, f)
    with open(os.path.join(model_destination, 'history.json'), 'wb') as f:
        json.dump(training_history, f)
    print("Done!")


