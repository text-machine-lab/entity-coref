import argparse
import time
import os
import sys
import numpy as np
import pickle
import json
import copy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from code.models import get_pre_ntm_model, MAXLEN
from code.build_data import build_dataFrame, DataGen, group_data


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("train_dir",
                        help="Directory containing training annotations")

    parser.add_argument("model_destination",
                        help="Where to store the trained model")

    parser.add_argument("--val_dir",
                        default=None,
                        help="Directory containing validation annotations")

    parser.add_argument("--no_ntm",
                        action='store_true',
                        default=False,
                        help="specify whether to use neural turing machine. default is to use ntm (no_ntm=false).")

    parser.add_argument("--neg_ratio",
                        default=0.8,
                        type=float,
                        help="negative cases ratio for downsampling. e.g. 0.5 means 50% instances are negative.")

    args = parser.parse_args()

    train_gen = DataGen(build_dataFrame(args.train_dir, threads=4))
    train_input_gen = train_gen.generate_input(negative_ratio=args.neg_ratio, file_batch=100)
    print("Loaded training data")

    if args.val_dir is not None:
        if os.path.isfile(os.path.join(args.val_dir, 'eval_data.pkl')):
            val_data_q = pickle.load(open(os.path.join(args.val_dir, 'eval_data.pkl')))
            evaluator = Evaluator(None, val_data_q)
        else:
            # Need the same word indexes and pos indexes for training and test data
            val_gen = DataGen(build_dataFrame(args.val_dir, threads=4), train_gen.word_indexes, train_gen.pos_tags)
            val_data_q = val_gen.generate_input(negative_ratio=None, looping=False).next()
            evaluator = Evaluator(None, val_data_q)
            evaluator.save_data(args.val_dir)
        print("Loaded validation data.")

    if args.no_ntm:
        model = get_pre_ntm_model(group_size=None, input_dropout=0.3, max_len=MAXLEN+10,
                                  embedding_matrix=train_gen.embedding_matrix, pos_size=len(train_gen.pos_tags))
        fit_batch_size = 1
        group_size = 40
    else:
        pass

    training_history = []
    for epoch in range(100):
        train_data_q = train_input_gen.next()
        n_training_files = len(train_data_q)
        epoch_history = []
        start = time.time()
        for n, data in enumerate(train_data_q):
            X, y = group_data(data, group_size)

            history = model.fit(X, y, batch_size=fit_batch_size, epochs=1, verbose=0, validation_split=0.2,
                            validation_data=None, shuffle=False, class_weight=None, sample_weight=None,
                            initial_epoch=0)
            if history.history:
                epoch_history.append(history.history)
            # reset states after a note file is processed
            model.reset_states()
            # if 'pairwise_out_accuracy' in history.history:
            #     main_affix = 'pairwise_out_'
            # else:
            #     main_affix = ''
            # try:
            #     acc = np.mean([h[main_affix + 'accuracy'][0] for h in epoch_history])
            # except KeyError:
            #     print("KeyError. accuracy is not found. Correct keys:", epoch_history[-1].keys())
            acc = np.mean([h['acc'] for h in epoch_history])
            loss = np.mean([h['loss'] for h in epoch_history])
            val_acc = np.mean([h['val_acc'] for h in epoch_history])
            val_loss = np.mean([h['val_loss'] for h in epoch_history])

            sys.stdout.write("epoch %d after training file %d/%d--- -%ds - loss : %.4f - acc : %.4f - val_loss : %.4f - val_acc : %.4f\r" % (
                epoch + 1, n + 1, n_training_files, int(time.time() - start), loss, acc, val_loss, val_acc))
            sys.stdout.flush()

        # if epoch > 2 and args.val_dir is not None:
        #     evaluator.model = model
        #     val_acc = evaluator.fast_eval()
        #     sys.stdout.write("epoch %d after training file %d/%d--- -%ds - loss : %.4f - acc : %.4f - val_acc : %.4f\r" % (
        #         epoch + 1, n + 1, n_training_files, int(time.time() - start), loss, acc, val_acc))
        sys.stdout.write("\n")
        training_history.append({'categorical_accuracy': acc, 'loss': loss})
        model.save(args.model_destination + 'final_model.h5')
    evaluator.model = model
    eval_results = evaluator.fast_eval()
    with open(os.path.join(args.model_destination, 'results.pkl'), 'w') as f:
        pickle.dump(eval_results, f)
    with open(os.path.join(args.model_destination, 'history.json'), 'w') as f:
        json.dump(training_history, f)


class Evaluator(object):
    def __init__(self, model, test_data_q):
        self.model = model
        self.test_data_q = test_data_q

    def save_data(self, path):
        with open(os.path.join(path, 'eval_data.pkl'), 'w') as f:
            pickle.dump(self.test_data_q, f)

    def fast_eval(self):
        Y_true = []
        Y_pred = []
        for data in self.test_data_q:
            X, y = group_data(data, 40)
            pred = self.model.predict(X, batch_size=10)
            Y_true.append(y)
            Y_pred.append(pred)

        Y_true = np.concatenate(Y_true)
        Y_true = Y_true.flatten()
        Y_pred = np.concatenate(Y_pred)
        Y_pred = Y_pred.flatten()
        Y_pred = np.round(Y_pred).astype(int)

        return precision_recall_fscore_support(Y_true, Y_pred)


if __name__ == "__main__":
    main()