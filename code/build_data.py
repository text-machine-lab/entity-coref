from __future__ import print_function
import subprocess
import os
import sys
import pandas as pd
import re
from collections import deque
import numpy as np
import time
from keras.preprocessing.sequence import pad_sequences
import multiprocessing

from word2vec import build_vocab
from models import EMBEDDING_DIM, MAXLEN

def build_dataFrame(path, threads=4):
    def worker(pid):
        print("worker %d started..." % pid)
        df = None
        counter = 0
        while not file_queue.empty():
            data_file = file_queue.get()
            # sys.stdout.write("Worker %d: %d files remained to be processed\r" % (pid, file_queue.qsize()))
            df = get_df(data_file, dataFrame=df)
            counter += 1
            if df is not None and counter % 10 == 0:
                data_queue.put(df)
                df = None
        if df is not None:
            data_queue.put(df)
        print("\nWorker %d closed." % pid)

    def worker_alive(workers):
        worker_alive = False
        for p in workers:
            if p.is_alive(): worker_alive = True
        return worker_alive

    assert os.path.isdir(path)
    cmd = 'find ' + path + ' -name "*gold_conll"'
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True,stdin=subprocess.PIPE)
    file_queue = multiprocessing.Queue()
    data_queue = multiprocessing.Queue(maxsize=10)
    for item in proc.stdout:
        file_queue.put(item.strip())
    n_files = file_queue.qsize()
    print('%d conll files found in %s' % (n_files, path))

    workers = [multiprocessing.Process(target=worker, args=(pid,)) for pid in range(threads)]

    for p in workers:
        p.daemon = True
        p.start()

    time.sleep(1)
    df = None

    while not data_queue.empty() or worker_alive(workers):
        item = data_queue.get()
        if df is None:
            df = item
        else:
            df = df.append(item)
        sys.stdout.write("Processed %d files from data queue\r" % len(df.doc_id.unique()))

    # Exit the completed processes
    print("\nFinished assembling data frame.")
    for p in workers:
        p.join()

    df.part_nb = pd.to_numeric(df.part_nb, errors='coerce')
    df.word_nb = pd.to_numeric(df.word_nb, errors='coerce')
    print("\ndata frame is built successfully!")
    print("Processed files: %d" % len(df.doc_id.unique()))

    return df


def get_df(data_file, dataFrame=None):
    data_list = []
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if line and line[0] != '#':
                fields = line.split()
                assert len(fields) >= 12
                fields = fields[:11] + [fields[-1]]
                data_list.append(fields)
                # df.loc[len(df)] = fields

    if not data_list:
        return None

    if dataFrame is None:
        # 12 columns, ignore predicate arguments
        columns = ['doc_id', 'part_nb', 'word_nb', 'word', 'pos', 'parse', 'predicate_lemma',
                   'predicate_frame', 'word_sense', 'speaker', 'name_entities', 'coref']
        dataFrame = pd.DataFrame(data_list, columns=columns)
    else:
        dataFrame = dataFrame.append(pd.DataFrame(data_list, columns=dataFrame.columns))

    return dataFrame


def get_entities(df):
    coref_entities = {}
    prefix = re.compile('\(\d+')
    suffix = re.compile('\d+\)')
    n = len(df)
    for i in range(n):
        coref = df.iloc[i].coref
        starts = prefix.findall(coref)
        ends = suffix.findall(coref)

        for item in starts:
            coref_id = df.iloc[i].doc_id + '-' + item[1:]
            if coref_id in coref_entities:
                coref_entities[coref_id]['start'].append(i)
            else:
                coref_entities[coref_id] = {'start': [i], 'end': []}

        for item in ends:
            coref_id = df.iloc[i].doc_id + '-' + item[:-1]
            assert coref_id in coref_entities
            coref_entities[coref_id]['end'].append(i)

    return coref_entities

class Entity(object):
    def __init__(self, coref_id, df, start_loc, end_loc):
        self.df = df
        self.start_loc = start_loc
        self.end_loc = end_loc
        self.coref_id = coref_id
        self.speaker = df.iloc[start_loc].speaker
        self.order = None

        self.get_words()
        self.get_pos_tags()
        self.get_context_representation()
        self.get_context_pos()

    def get_words(self):
        self.words = [self.df.iloc[i].word for i in range(self.start_loc, self.end_loc+1)]

    def get_pos_tags(self):
        self.pos_tags = [self.df.iloc[i].pos for i in range(self.start_loc, self.end_loc + 1)]

    def get_order(self, coref_entities, locations=None):
        """get the order of the entity in a doc  e.g. 5 means it is the 5th entity"""
        if locations is not None:
            self.order = locations.index(self.start_loc)
            return self.order, locations

        doc_id = self.df.iloc[self.start_loc].doc_id
        # doc_entries = self.df.loc[self.df.doc_id == doc_id]
        locations = []
        for coref_id in coref_entities:
            if doc_id in coref_id: # strings match
                locations += coref_entities[coref_id]['start']
        locations.sort()
        self.order = locations.index(self.start_loc)
        return self.order, locations

    def get_context_representation(self):
        left_edge = max(0, self.start_loc-5)
        left_words = [self.df.iloc[i].word for i in range(left_edge, self.start_loc)]
        right_edge = min(len(self.df), self.end_loc+6)
        right_words = [self.df.iloc[i].word for i in range(self.end_loc+1, right_edge)]

        self.context_words =  left_words + ['_START_'] + self.words + ['_END_'] + right_words

    def get_context_pos(self):
        left_edge = max(0, self.start_loc - 5)
        left_pos = [self.df.iloc[i].pos for i in range(left_edge, self.start_loc)]
        right_edge = min(len(self.df), self.end_loc + 6)
        right_pos = [self.df.iloc[i].pos for i in range(self.end_loc + 1, right_edge)]

        self.context_pos = left_pos + ['_START_POS_'] + self.pos_tags + ['_END_POS_'] + right_pos


class DataGen(object):
    def __init__(self, df, word_indexes={}, pos_tags=[]):
        self.df = df
        self.word_indexes = word_indexes
        self.pos_tags = pos_tags
        if not self.word_indexes:
            self.get_embedding_matrix()
        if not self.pos_tags:
            self.get_pos_tags()

    def generate_input(self, negative_ratio=0.8, file_batch=100, looping=True):
        """Generate pairwise input
           negative_ratio = # negative / # total
        """
        assert negative_ratio is None or negative_ratio < 1
        doc_ids = self.df.doc_id.unique()
        data_q = deque()

        while True:
            np.random.shuffle(doc_ids)
            for doc_id in doc_ids:
                # print("Generating data for %s" % doc_id)
                doc_df = self.df.loc[self.df.doc_id == doc_id]
                doc_coref_entities = get_entities(doc_df)

                # get entity list
                entities = []
                locations = None
                for coref_id in doc_coref_entities:
                    for start_loc, end_loc in zip(doc_coref_entities[coref_id]['start'], doc_coref_entities[coref_id]['end']):
                        entity = Entity(coref_id, doc_df, start_loc, end_loc)
                        order, locations = entity.get_order(doc_coref_entities, locations=locations)
                        entities.append((order, entity))

                if not entities:
                    continue
                entities = [e[1] for e in sorted(entities, key=lambda x: x[0])]

                # generate pairwise input. Process in narrative order
                X0 = []
                X1 = []
                X2 = []
                X3 = []
                X4 = []
                Y = []
                for i, entity in enumerate(entities):
                    for j in range(0, i):
                        distance = entity.order - entities[j].order
                        words_i = entity.context_words
                        word_i_indexes = [self.word_indexes[word] if word in self.word_indexes else self.word_indexes['UKN'] for word in words_i]
                        pos_i = entity.context_pos
                        pos_i_indexes = [self.pos_tags.index(pos) if pos in self.pos_tags else self.pos_tags.index('UKN') for pos in pos_i]
                        words_j = entities[j].context_words
                        word_j_indexes = [self.word_indexes[word] if word in self.word_indexes else self.word_indexes['UKN'] for word in words_j]
                        pos_j = entities[j].context_pos
                        pos_j_indexes = [self.pos_tags.index(pos) if pos in self.pos_tags else self.pos_tags.index('UKN') for pos in pos_j]

                        X0.append([distance]) # use list to add a dimension
                        X1.append(word_i_indexes)
                        X2.append(pos_i_indexes)
                        X3.append(word_j_indexes)
                        X4.append(pos_j_indexes)
                        if entity.coref_id == entities[j].coref_id:
                            Y.append(1)
                        else:
                            Y.append(0)

                X0 = np.array(X0)
                X1 = pad_sequences(X1, maxlen=MAXLEN + 10, dtype='int32', padding='pre', truncating='post', value=0)
                X2 = pad_sequences(X2, maxlen=MAXLEN + 10, dtype='int32', padding='pre', truncating='post', value=0)
                X3 = pad_sequences(X3, maxlen=MAXLEN + 10, dtype='int32', padding='pre', truncating='post', value=0)
                X4 = pad_sequences(X4, maxlen=MAXLEN + 10, dtype='int32', padding='pre', truncating='post', value=0)
                Y = np.array(Y)

                datum = [[X0, X1, X2, X3, X4], Y]

                if negative_ratio is not None: # perform downsampling
                    neg_case_indexes = np.where(Y == 0)[0]
                    pos_case_indexes = np.where(Y == 1)[0]
                    np.random.shuffle(neg_case_indexes)
                    n_neg_samples = min(len(neg_case_indexes), int(len(pos_case_indexes)*negative_ratio/(1 - negative_ratio)))
                    neg_case_indexes = neg_case_indexes[0:n_neg_samples]

                    training_indexes = np.concatenate([pos_case_indexes, neg_case_indexes])
                    training_indexes.sort()

                    for i, item in enumerate(datum[0]):
                        datum[0][i] = item[training_indexes]

                    datum[1] = datum[1][training_indexes] # Y

                data_q.append(datum)
                if looping and len(data_q) == file_batch:
                    yield data_q
                    data_q = deque()

            if not looping:  # yield the whole data set, and break
                yield data_q
                break

    def get_embedding_matrix(self, word_vectors=None):
        if word_vectors is None:
            print('Loading word embeddings...')
            glove_path = os.environ["TEA_PATH"] + 'embeddings/glove.840B.300d.txt'
            word_vectors = build_vocab(self.df.word.unique(), glove_path, K=200000)
            word_vectors['_START_'] = np.ones(EMBEDDING_DIM)
            # word_vectors['_START_POS_'] = np.ones(EMBEDDING_DIM)
            word_vectors['_END_'] = - np.ones(EMBEDDING_DIM)
            # word_vectors['_END_POS_'] = - np.ones(EMBEDDING_DIM)
            word_vectors['UKN'] = np.random.uniform(-0.5, 0.5, EMBEDDING_DIM)

        word_indexes = {}
        embedding_matrix = np.random.uniform(low=-0.5, high=0.5, size=(len(word_vectors) + 1, EMBEDDING_DIM))
        for index, word in enumerate(sorted(word_vectors.keys())):
            word_indexes[word] = index + 1
            embedding_vector = word_vectors.get(word, None)
            embedding_matrix[index + 1] = embedding_vector
        embedding_matrix[0] = np.zeros(EMBEDDING_DIM)  # used for mask/padding

        self.embedding_matrix = embedding_matrix
        self.word_indexes = word_indexes

    def get_pos_tags(self):
        all_pos_tags = self.df.pos.unique()
        all_pos_tags.sort()
        print("%d pos tags found" % len(all_pos_tags))
        print(all_pos_tags)
        self.pos_tags = np.append(all_pos_tags, ['_START_POS_', '_END_POS_', 'UKN']).tolist()

def slice_data(data, group_size):
    """Slice data to equal size"""
    X, y = data
    if group_size == 0 or group_size is None: # special case, only one chunk
        yield X, y
    else:
        n = len(y)
        n_chunks = n / group_size

        if n_chunks > 0:
            for m in range(n_chunks):
                X_out = [x[m*group_size: (m+1)*group_size] for x in X]
                y_out = y[m*group_size: (m+1)*group_size]
                yield X_out, y_out

        leftover = n % group_size
        if leftover > 0:
            to_add = group_size - leftover
            indexes_to_add = np.random.choice(n, to_add)  # randomly sample more instances
            indexes = np.concatenate((np.arange(n_chunks * group_size, n), indexes_to_add))
            X_out = [x[indexes] for x in X]
            y_out = y[indexes]
            yield X_out, y_out

def group_data(data, group_size):
    X_out = None
    y_out = None
    for slice in slice_data(data, group_size):
        X, y = slice
        X = [np.expand_dims(x, axis=0) for x in X]
        y = np.expand_dims(y, axis=0)
        if X_out is None:
            X_out = X
            y_out = y
        else:
            X_out = [np.concatenate((X_out[i], X[i]), axis=0) for i in range(len(X))]
            y_out = np.concatenate((y_out, y), axis=0)
    # make y 3D (batch, group, 1)
    return X_out, np.expand_dims(y_out, axis=-1)
