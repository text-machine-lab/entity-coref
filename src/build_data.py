from __future__ import print_function
import subprocess
import os
import sys
import pandas as pd
import re
from collections import deque
import numpy as np
import time
import multiprocessing
from itertools import combinations

from src.word2vec import build_vocab

EMBEDDING_DIM = 300
NEIGHBORHOOD = 3  # minimum distance between entities
MAX_DISTANCE = 15  #40
MAXLEN = 10 + 10  # max length of entities allowed (target length + context length)
BATCH_SIZE = 5

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This is copied from keras.preprocessing.sequence.pad_sequences
    We put it here so we do not have to import keras, which depends on tensorflow.

    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)


    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def build_dataFrame(path, threads=4, suffix='gold_conll'):
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
    cmd = 'find ' + path + ' -name "*%s"' % suffix
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

    while df is None or len(df.doc_id.unique()) < n_files:
        item = data_queue.get()
        if df is None:
            df = item
        else:
            df = df.append(item)
        sys.stdout.write("Processed %d files from data queue\r" % len(df.doc_id.unique()))
        # if not worker_alive(workers): break

    time.sleep(1)
    assert data_queue.empty()
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
        left_edge = max(0, self.start_loc - 5)
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

    def generate_input(self, negative_ratio=0.8, file_batch=100, looping=True, test_data=False, **kwargs):
        """Generate pairwise input
           negative_ratio = # negative / # total
           file_batch: # files to process and yield each time
        """
        assert negative_ratio is None or negative_ratio < 1
        if 'max_distance' in kwargs:
            max_distance = kwargs['max_distance']
        else:
            max_distance = MAX_DISTANCE

        doc_ids = self.df.doc_id.unique()
        data_q = deque()
        if test_data:
            file_batch = 1 # yield data from one file only
            negative_ratio = None
        while True:
            if not test_data: np.random.shuffle(doc_ids)
            for doc_id in doc_ids:
                index_map = {}
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
                entities = [e[1] for e in sorted(entities, key=lambda x: x[0])]  # sorted according to order
                index = 0

                # generate pairwise input. Process in narrative order
                X0 = []
                X1 = []
                X2 = []
                X3 = []
                X4 = []
                X5 = []
                Y = []
                for i, entity in enumerate(entities):
                    for j in range(0, i):
                        distance = entity.order - entities[j].order
                        if distance > max_distance:
                            continue
                        if entity.speaker == entities[j].speaker:
                            same_speaker = 1
                        else:
                            same_speaker = 0
                        words_i = entity.context_words
                        # words_i = entity.words
                        word_i_indexes = [self.word_indexes[word] if word in self.word_indexes else self.word_indexes['UKN'] for word in words_i]
                        pos_i = entity.context_pos
                        # pos_i = entity.pos_tags
                        pos_i_indexes = [self.pos_tags.index(pos)+1 if pos in self.pos_tags else self.pos_tags.index('UKN') for pos in pos_i]
                        words_j = entities[j].context_words
                        # words_j = entities[j].words
                        word_j_indexes = [self.word_indexes[word] if word in self.word_indexes else self.word_indexes['UKN'] for word in words_j]
                        pos_j = entities[j].context_pos
                        # pos_j = entities[j].pos_tags
                        pos_j_indexes = [self.pos_tags.index(pos)+1 if pos in self.pos_tags else self.pos_tags.index('UKN') for pos in pos_j]
                        y = int(entity.coref_id == entities[j].coref_id)

                        X0.append([distance]) # use list to add a dimension
                        X1.append([same_speaker])
                        X2.append(word_i_indexes)
                        X3.append(pos_i_indexes)
                        X4.append(word_j_indexes)
                        X5.append(pos_j_indexes)
                        Y.append(y)
                        if test_data:
                            index_map[(doc_id, (entity.start_loc, entity.end_loc), (entities[j].start_loc, entities[j].end_loc))] = index
                        index += 1

                X0 = np.array(X0)
                X1 = np.array(X1)
                X2 = pad_sequences(X2, maxlen=MAXLEN, dtype='int32', padding='pre', truncating='post', value=0)
                X3 = pad_sequences(X3, maxlen=MAXLEN, dtype='int32', padding='pre', truncating='post', value=0)
                X4 = pad_sequences(X4, maxlen=MAXLEN, dtype='int32', padding='pre', truncating='post', value=0)
                X5 = pad_sequences(X5, maxlen=MAXLEN, dtype='int32', padding='pre', truncating='post', value=0)
                Y = np.array(Y)

                if test_data:
                    datum = [[X0, X1, X2, X3, X4, X5], Y, index_map]
                else:
                    datum = [[X0, X1, X2, X3, X4, X5], Y]

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


    def generate_triad_input(self, file_batch=100, looping=True, test_data=False, threads=4, **kwargs):
        """Generate triad input
           file_batch: # files to process and yield each time
         """
        if 'max_distance' in kwargs:
            max_distance = kwargs['max_distance']
        else:
            max_distance = MAX_DISTANCE
        print("max distance: %d" % max_distance)
        def worker(doc_id_q, out_q):
            while True:
                doc_id = doc_id_q.get()
                index_map = {}
                # print("Generating data for %s" % doc_id)
                doc_df = self.df.loc[self.df.doc_id == doc_id]
                doc_coref_entities = get_entities(doc_df)

                # get entity list
                entities = []
                locations = None
                for coref_id in doc_coref_entities:
                    for start_loc, end_loc in zip(doc_coref_entities[coref_id]['start'],
                                                  doc_coref_entities[coref_id]['end']):
                        entity = Entity(coref_id, doc_df, start_loc, end_loc)
                        order, locations = entity.get_order(doc_coref_entities, locations=locations)
                        entities.append((order, entity))

                if not entities:
                    continue
                entities = [e[1] for e in sorted(entities, key=lambda x: x[0])]  # sorted according to order
                N = len(entities)
                if N < 2:
                    print("Only one entity in %s" % doc_id)
                    continue
                elif N < 3:
                    entities.append(entities[0])  # expand a dyad to a triad
                    N += 1

                triad_indexes = combinations(range(N), 3)

                X = [[] for _ in range(12)]
                Y = []
                index = 0
                for a, b, c in triad_indexes:
                    triad = (entities[a], entities[b], entities[c])
                    distances = DataGen.get_triad_distances(triad)
                    diameter = max([abs(item) for item in distances])  # maximum distance between entities
                    neighborhood = min([abs(item) for item in distances])  # minimum distance between entities
                    if diameter <= max_distance and neighborhood <= NEIGHBORHOOD:

                        speaker_identities = [int(triad[0].speaker == triad[1].speaker),
                                              int(triad[1].speaker == triad[2].speaker),
                                              int(triad[2].speaker == triad[0].speaker)]

                        word_indexes = [self.get_word_indexes(triad[0].context_words),
                                        self.get_word_indexes(triad[1].context_words),
                                        self.get_word_indexes(triad[2].context_words)]

                        pos_indexes = [self.get_pos_indexes(triad[0].context_pos),
                                       self.get_pos_indexes(triad[1].context_pos),
                                       self.get_pos_indexes(triad[2].context_pos)]

                        X_triad = distances + speaker_identities + word_indexes + pos_indexes
                        # locations = [triad[ind].order for ind in range(3)]
                        # X_triad = locations + speaker_identities + word_indexes + pos_indexes

                        for i in range(12):
                            X[i].append(X_triad[i])
                        y_triad = [int(triad[0].coref_id == triad[1].coref_id),
                                   int(triad[1].coref_id == triad[2].coref_id),
                                   int(triad[2].coref_id == triad[0].coref_id)]
                        Y.append(np.array(y_triad))

                        if test_data:
                            index_map[(doc_id,
                                       (triad[0].start_loc, triad[0].end_loc),
                                       (triad[1].start_loc, triad[1].end_loc),
                                       (triad[2].start_loc, triad[2].end_loc))] = index
                            index += 1

                for i in range(6):  # distance and speaker
                    X[i] = np.array(X[i])
                    X[i] = np.expand_dims(X[i], axis=-1)

                for i in range(6, 12):  # word and pos tag indexes
                    X[i] = pad_sequences(X[i], maxlen=MAXLEN, dtype='int32', padding='pre', truncating='post',
                                         value=0)

                Y = np.array(Y)

                if test_data:
                    datum = [X, Y, index_map]
                else:
                    datum = [X, Y]

                out_q.put(datum)

        # main process
        doc_ids = self.df.doc_id.unique()
        data_q = deque()
        if test_data:
            file_batch = 1  # yield data from one file each time
        out_q = multiprocessing.Queue(maxsize=200)
        doc_id_q = multiprocessing.Queue()
        workers = [multiprocessing.Process(target=worker, args=(doc_id_q, out_q)) for _ in range (threads)]
        for worker in workers:
            worker.daemon = True
            worker.start()

        while True:
            if not test_data:
                np.random.shuffle(doc_ids)
            if doc_id_q.empty():
                for doc_id in doc_ids:
                    doc_id_q.put(doc_id)

            datum = out_q.get()
            data_q.append(datum)
            if looping and len(data_q) == file_batch:
                yield data_q
                data_q = deque()

            if not looping and len(data_q) == len(doc_ids):  # yield the whole data set, and break
                yield data_q
                break

    @staticmethod
    def get_triad_distances(triad):
        d0 = triad[0].order - triad[1].order
        d1 = triad[1].order - triad[2].order
        d2 = triad[2].order - triad[0].order
        return [d0, d1, d2]

    def get_word_indexes(self, word_list):
        return [self.word_indexes[word] if word in self.word_indexes else self.word_indexes['UKN'] for word in word_list]

    def get_pos_indexes(self, pos_list):
        return [self.pos_tags.index(pos) + 1 if pos in self.pos_tags else self.pos_tags.index('UKN') for pos in pos_list]

    def get_embedding_matrix(self, word_vectors=None):
        if word_vectors is None:
            print('Loading word embeddings...')
            glove_path = os.environ['HOME'] + '/projects/embeddings/glove.840B.300d.txt'
            word_vectors = build_vocab(self.df.word.unique(), glove_path, K=200000)
            word_vectors['_START_'] = np.ones(EMBEDDING_DIM)
            word_vectors['_END_'] = - np.ones(EMBEDDING_DIM)
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
    """Slice data to equal size
        group_size: # instances to yield each time. If 0 or None, yield all
    """
    X, y = data
    if len(y.shape) == 1:
        y = np.expand_dims(y, axis=-1)  # make y 2D (group, 1)

    if group_size == 0 or group_size is None: # special case, only one chunk
        yield X, y
    else:
        n = len(y)
        n_chunks = int(n / group_size)

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

def group_data(data, group_size, batch_size=None):
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

        if batch_size is not None and batch_size == y_out.shape[0]:
            yield X_out, y_out
            X_out = None
            y_out = None

    if batch_size is None: # a yield for a file, whatever size
        yield X_out, y_out

    # make batch full
    elif y_out is not None:
        to_add = batch_size - y_out.shape[0]
        for _ in range(to_add):
            X_out = [np.concatenate([X_out[i], item]) for i, item in enumerate(X)]
            y_out = np.concatenate([y_out, y])
        yield X_out, y_out
