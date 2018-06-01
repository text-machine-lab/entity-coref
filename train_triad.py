import argparse
import time
import os
import sys
import numpy as np
import pickle
import json
import copy
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
import multiprocessing

from src.models import MAXLEN, BATCH_SIZE, get_triad_model
from src.build_data import build_dataFrame, DataGen, group_data, slice_data
from predict import TriadEvaluator


def fill_data_q(input_gen, subproc_queue):
    while True:
        if not subproc_queue.full():
            data_q = next(input_gen)
            subproc_queue.put(data_q)
        else:
            time.sleep(1)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("train_dir",
                        help="Directory containing training annotations")

    parser.add_argument("model_destination",
                        help="Where to store the trained model")

    parser.add_argument("--val_dir",
                        default=None,
                        help="Directory containing validation annotations")

    # parser.add_argument("--no_ntm",
    #                     action='store_true',
    #                     default=False,
    #                     help="specify whether to use neural turing machine. default is to use ntm (no_ntm=false).")

    parser.add_argument("--neg_ratio",
                        default=0.8,
                        type=float,
                        help="negative cases ratio for downsampling. e.g. 0.5 means 50% instances are negative.")

    parser.add_argument("--load_model",
                        action='store_true',
                        default=False,
                        help="Load saved model and resume training from there")

    parser.add_argument("--epochs",
                        default=200,
                        type=int,
                        help="Load saved model and resume training from there")

    args = parser.parse_args()

    train_gen = DataGen(build_dataFrame(args.train_dir, threads=3))
    train_input_gen = train_gen.generate_triad_input(file_batch=50)
    print("Loaded training data")
    with open(os.path.join(args.model_destination, 'word_indexes.pkl'), 'wb') as f:
        pickle.dump(train_gen.word_indexes, f)
    with open(os.path.join(args.model_destination, 'pos_tags.pkl'), 'wb') as f:
        pickle.dump(train_gen.pos_tags, f)

    group_size = 40

    if args.load_model:
        model = load_model(os.path.join(args.model_destination, 'model.h5'))
    else:
        model = get_triad_model(group_size=group_size, input_dropout=0.5, max_len=MAXLEN,
                                  embedding_matrix=train_gen.embedding_matrix, pos_size=len(train_gen.pos_tags))

    training_history = []
    evaluator = None

    if args.val_dir is not None:
        # Need the same word indexes and pos indexes for training and test data
        val_gen = DataGen(build_dataFrame(args.val_dir, threads=1), train_gen.word_indexes, train_gen.pos_tags)
        val_input_gen = val_gen.generate_triad_input(file_batch=10, looping=True, threads=1)
        # just get data from 1 for try
        val_data_q = next(val_input_gen)
        val_data = val_data_q[0]
        # val_X, val_y = next(group_data(val_data, group_size, batch_size=None))
        val_X, val_y = val_data
        validation_split = 0
    else:
        validation_split = 0.2

    # subproc_queue = multiprocessing.Queue(maxsize=100)
    # subproc = multiprocessing.Process(target=fill_data_q, args=(train_input_gen, subproc_queue))
    # subproc.daemon = True
    # subproc.start()

    for epoch in range(args.epochs):
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
                            initial_epoch=0)#, callbacks=[tensorboard])
            if history.history:
                epoch_history.append(history.history)

            acc = np.mean([h['acc'] for h in epoch_history])
            loss = np.mean([h['loss'] for h in epoch_history])
            val_acc = np.mean([h['val_acc'] for h in epoch_history])
            val_loss = np.mean([h['val_loss'] for h in epoch_history])

            sys.stdout.write("epoch %d after training file %d/%d--- -%ds - loss : %.4f - acc : %.4f - val_loss : %.4f - val_acc : %.4f\r" % (
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
                model.save(os.path.join(args.model_destination, 'model.h5'))
            except:
                model.save_weights(os.path.join(args.model_destination, 'weights.h5'))

            if args.val_dir:
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
    with open(os.path.join(args.model_destination, 'history.json'), 'wb') as f:
        json.dump(training_history, f)
    print("Done!")


if __name__ == "__main__":
    main()