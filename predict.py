import argparse
import time
import os
import multiprocessing
import sys
import numpy as np
import pickle
import shutil
import glob
from collections import defaultdict
from sklearn.metrics import classification_report, precision_recall_fscore_support

from src.build_data import build_dataFrame, DataGen, group_data, slice_data
from src.models import clustering, BATCH_SIZE

class Evaluator(object):
    def __init__(self, model, test_data_q):
        self.model = model
        self.test_data_q = test_data_q

    def fast_eval(self, test_data=False):
        Y_true = []
        Y_pred = []
        for data in self.test_data_q:
            if len(data) == 3:
                data = data[:2]
            for X, y in group_data(data, 40, BATCH_SIZE):
                pred = self.model.predict_on_batch(X)
                Y_true.append(y)
                Y_pred.append(pred)

        Y_true = np.concatenate(Y_true)
        Y_true = Y_true.flatten()
        Y_pred = np.concatenate(Y_pred)
        Y_pred = Y_pred.flatten()
        Y_pred = np.round(Y_pred).astype(int)

        return classification_report(Y_true, Y_pred, digits=3)

    def write_results(self, df, dest_path):
        n_files = len(self.test_data_q)
        print("# files: %d" % n_files)
        for i, data in enumerate(self.test_data_q):
            if i == n_files:
                break
            X, y, index_map = data
            pred = []
            for X, y in group_data([X, y], 40, batch_size=BATCH_SIZE):
                pred.append(self.model.predict_on_batch(X))
            pred = np.concatenate(pred)
            pred = pred.flatten()

            pair_results = {}
            for key in index_map:
                pair_results[(key[1], key[2])] = pred[index_map[key]]
            locs, clusters, _ = clustering(pair_results)
            doc_id = key[0]
            length = len(df.loc[df.doc_id == doc_id])

            # print("Saving %s results..." % doc_id)
            sys.stdout.write("Saving results %d / %d\r" % (i + 1, n_files))
            sys.stdout.flush()
            corefs = ['-' for i in range(length)]
            for loc, cluster in zip(locs, clusters):
                start, end = loc
                if corefs[start] == '-':
                    corefs[start] = '(' + str(cluster)
                else:
                    corefs[start] += '|(' + str(cluster)

                if corefs[end] == '-':
                    corefs[end] = str(cluster) + ')'
                elif start == end:
                    corefs[end] += ')'
                else:
                    corefs[end] += '|' + str(cluster) + ')'
            with open(os.path.join(dest_path, doc_id.split('/')[-1]), 'w') as f:
                f.write('#begin document (%s);\n' % doc_id)
                for coref in corefs:
                    f.write(doc_id + '\t' + coref +'\n')
                f.write('\n#end document\n')
        print("Completed saving results!")


class TriadEvaluator(object):
    def __init__(self, model, test_input_gen, file_batch=10):
        self.model = model
        # self.test_data_gen = test_data_gen
        # self.test_input_gen = test_data_gen.generate_triad_input(file_batch=file_batch, threads=1)
        self.test_input_gen = test_input_gen
        self.data_q_store = multiprocessing.Queue(maxsize=5)
        # self.data_available = False

    # def fill_q_store(self):
    #     print("evaluator data filler started...")
    #     self.data_available = True
    #     while True:
    #         if not self.data_q_store.full():
    #             self.data_q_store.put(self.test_input_gen.next())
    #         else:
    #             time.sleep(1)
    #     self.data_available = False

    def fast_eval(self):
        """Fast evaluation from a subset of test files
           Scores are based on pairs only
        """
        # assert self.data_available
        Y_true = []
        Y_pred = []
        # test_data_q = self.data_q_store.get()
        test_data_q = next(self.test_input_gen)
        for data in test_data_q:
            if len(data) == 3:
                data = data[:2]
            for X, y in slice_data(data, 100):
                pred = self.model.predict(X) # (group_size, 3)
                Y_true.append(y)
                Y_pred.append(pred)

        Y_true = np.concatenate(Y_true)
        Y_true = Y_true.flatten()
        Y_pred = np.concatenate(Y_pred)
        Y_pred = Y_pred.flatten()
        Y_pred = np.round(Y_pred).astype(int)

        return classification_report(Y_true, Y_pred, digits=3)

    def write_results(self, df, dest_path, n_iterations, save_dendrograms=True):
        """Perform evaluation on all test data, write results"""
        # assert self.data_available
        print("# files: %d" % n_iterations)

        all_pairs_true = []
        all_pairs_pred = []
        processed_docs = set([])
        discarded = 0
        for i in range(n_iterations):
            # test_data_q = self.data_q_store.get()
            test_data_q = next(self.test_input_gen)
            assert len(test_data_q) == 1  # only process one file
            X, y, index_map = test_data_q[0]
            doc_id = list(index_map.keys())[0][0]  # python3 does not support keys() as a list
            if doc_id in processed_docs:
                print("%s already processed before!" % doc_id)
                continue
            processed_docs.add(doc_id)

            pred = []
            for X, _ in slice_data([X, y], 50):  # do this to avoid very large batches
                pred.append(self.model.predict(X))
            pred = np.concatenate(pred)
            pred = np.reshape(pred, [-1, 3])  # in case there are batches

            true = np.reshape(y, [-1, 3])

            pair_results = defaultdict(list)
            pair_true = {}
            for key in index_map:
                if sum(np.round(pred[index_map[key]])) == 2:  # skip illogical triads
                    discarded += 1
                pair_results[(key[1], key[2])].append(pred[index_map[key]][0])
                pair_results[(key[2], key[3])].append(pred[index_map[key]][1])
                pair_results[(key[3], key[1])].append(pred[index_map[key]][2])

                pair_true[(key[1], key[2])] = true[index_map[key]][0]
                pair_true[(key[2], key[3])] = true[index_map[key]][1]
                pair_true[(key[3], key[1])] = true[index_map[key]][2]

            pair_results_mean = {}
            for key, value in pair_results.items():
                # mean_value = TriadEvaluator.nonlinear_mean(value)
                mean_value = TriadEvaluator.top_n_mean(value, 1)
                pair_results_mean[key] = mean_value
                all_pairs_pred.append(mean_value)
                all_pairs_true.append(pair_true[key])

            locs, clusters, linkage = clustering(pair_results_mean, binarize=False)
            _, clusters_true, linkage_true = clustering(pair_true, binarize=False)
            if save_dendrograms:
                np.save(os.path.join(dest_path, 'linkages', doc_id.split('/')[-1]+'.npy'), linkage)
                np.save(os.path.join(dest_path, 'true-linkages', doc_id.split('/')[-1] + '.npy'), linkage_true)

            length = len(df.loc[df.doc_id == doc_id])
            # print("Saving %s results..." % doc_id)
            sys.stdout.write("Saving results %d / %d\r" % (i + 1, n_iterations))
            sys.stdout.flush()
            corefs = ['-' for _ in range(length)]
            for loc, cluster in zip(locs, clusters):
                start, end = loc
                if corefs[start] == '-':
                    corefs[start] = '(' + str(cluster)
                else:
                    corefs[start] += '|(' + str(cluster)

                if corefs[end] == '-':
                    corefs[end] = str(cluster) + ')'
                elif start == end:
                    corefs[end] += ')'
                else:
                    corefs[end] += '|' + str(cluster) + ')'
            with open(os.path.join(dest_path, 'responses', doc_id.split('/')[-1]), 'w') as f:
                f.write('#begin document (%s);\n' % doc_id)
                for coref in corefs:
                    f.write(doc_id + '\t' + coref + '\n')
                f.write('\n#end document\n')

        print("Completed saving results!")
        print("Pairwise evaluation:")
        print("True histogram", np.histogram(all_pairs_true, bins=4))
        print("Prediction histogram", np.histogram(all_pairs_pred, bins=4))
        print(classification_report(all_pairs_true, np.round(all_pairs_pred), digits=3))
        print("Discarded triads:", discarded)

    @staticmethod
    def top_n_mean(values, n):
        values.sort(reverse=True)
        if len(values) >= n:
            values = values[:n]
        return np.mean(values)

    @staticmethod
    def median(values):
        values.sort(reverse=True)
        return values[len(values)/2]

    @staticmethod
    def nonlinear_mean(values):
        return np.mean(np.round(values))

def scorer(path=None):
    if path is None:
        path = './scorers/v8.01/results/dev/'
    if path[-1] != '/': path += '/'

    keys = glob.glob(path+'keys/*')
    responses = glob.glob(path+'responses/*')
    combined_key = path + 'key.tmp'
    combined_response = path + 'response.tmp'

    with open(combined_key, 'wb') as f_key:
        for filename in keys:
            with open(filename, 'rb') as readfile:
                shutil.copyfileobj(readfile, f_key)

    with open(combined_response, 'wb') as f_response:
        for filename in responses:
            with open(filename, 'rb') as readfile:
                shutil.copyfileobj(readfile, f_response)

    # cmd = 'perl scorers/v8.01/scorer.pl all ' + combined_key + ' ' + combined_response + ' none'
    # proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, stdin=subprocess.PIPE)
    # print(proc.stdout)

def main():
    # from keras.models import load_model


    parser = argparse.ArgumentParser()

    parser.add_argument("model_dir",
                        help="Directory containing the trained model")

    parser.add_argument("test_dir",
                        help="Directory containing test files")

    parser.add_argument("result_dir",
                        help="Directory to write result files")

    parser.add_argument("--score_only",
                        action='store_true',
                        default=False,
                        help="Perform evaluation only")

    parser.add_argument("--triad",
                        action='store_true',
                        default=False,
                        help="use triads")

    args = parser.parse_args()

    if not args.score_only:
        with open(os.path.join(args.model_dir, 'word_indexes.pkl'), 'rb') as f:
            word_indexes = pickle.load(f)
        with open(os.path.join(args.model_dir, 'pos_tags.pkl'), 'rb') as f:
            pos_tags = pickle.load(f)

        df = build_dataFrame(args.test_dir, threads=1, suffix='auto_conll')
        # model = load_model(os.path.join(args.model_dir, 'model.h5'))
        model = torch.load(os.path.join(args.model_dir, 'model.pt'))
        model.eval()
        print("Loaded model")

        if args.triad:
            n_files = len(df.doc_id.unique())
            test_gen = DataGen(df, word_indexes, pos_tags)
            test_input_gen = test_gen.generate_triad_input(looping=True, test_data=True, threads=4)
            evaluator = TriadEvaluator(model, test_input_gen)

            # evaluator.data_q_store = multiprocessing.Queue(maxsize=200)
            # filler = multiprocessing.Process(target=evaluator.fill_q_store, args=())
            # filler.start()
            # evaluator.data_available = True
            evaluator.write_results(df, args.result_dir, n_iterations=n_files)
            # filler.terminate()  # we cannot use daemons because filler has children

        else:
            test_gen = DataGen(df, word_indexes, pos_tags)
            test_data_q = next(test_gen.generate_input(looping=False, test_data=True))
            print("Loaded test data")

            evaluator = Evaluator(model, test_data_q)
            print("Performing fast evaluation...")
            print(evaluator.fast_eval())
            print("Saving result files...")
            evaluator.write_results(df, args.result_dir)

    scorer('./scorers/v8.01/results/test/')

if __name__ == "__main__":
    import torch
    from torch_models import CorefTagger
    main()
