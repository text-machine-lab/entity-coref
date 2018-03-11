import argparse
import time
import os
import sys
import numpy as np
import pickle
import json
import copy
import tensorflow as tf
import keras
from keras.models import load_model
from keras.callbacks import TensorBoard

from code.models import get_pre_ntm_model2, MAXLEN, get_combined_ntm_model, BATCH_SIZE
from code.build_data import build_dataFrame, DataGen, group_data
from predict import Evaluator


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

    parser.add_argument("--load_model",
                        action='store_true',
                        default=False,
                        help="Load saved model and resume training from there")

    parser.add_argument("--epochs",
                        default=200,
                        type=int,
                        help="Load saved model and resume training from there")

    args = parser.parse_args()

    train_gen = DataGen(build_dataFrame(args.train_dir, threads=4))
    train_input_gen = train_gen.generate_input(negative_ratio=args.neg_ratio, file_batch=100)
    print("Loaded training data")
    with open(os.path.join(args.model_destination, 'word_indexes.pkl'), 'w') as f:
        pickle.dump(train_gen.word_indexes, f)
    with open(os.path.join(args.model_destination, 'pos_tags.pkl'), 'w') as f:
        pickle.dump(train_gen.pos_tags, f)

    if args.val_dir is not None:
        if os.path.isfile(os.path.join(args.val_dir, 'eval_data.pkl')):
            val_data_q = pickle.load(open(os.path.join(args.val_dir, 'eval_data.pkl')))
        else:
            # Need the same word indexes and pos indexes for training and test data
            val_gen = DataGen(build_dataFrame(args.val_dir, threads=4), train_gen.word_indexes, train_gen.pos_tags)
            val_data_q = val_gen.generate_input(negative_ratio=None, looping=False).next()
            with open(os.path.join(args.val_dir, 'eval_data.pkl'), 'w') as f:
                pickle.dump(val_data_q, f)
        print("Loaded validation data.")

    group_size = 40

    if args.load_model:
        model = load_model(os.path.join(args.model_destination, 'final_model.h5'))
    elif args.no_ntm:
        model = get_pre_ntm_model2(group_size=group_size, input_dropout=0.3, max_len=MAXLEN,
                                  embedding_matrix=train_gen.embedding_matrix, pos_size=len(train_gen.pos_tags))
    else:
        model = get_combined_ntm_model(batch_size=BATCH_SIZE, group_size=group_size, input_dropout=0.3, max_len=MAXLEN, embedding_matrix=train_gen.embedding_matrix,
                                       pos_size=len(train_gen.pos_tags), m_depth=256, n_slots=128, ntm_output_dim=128,
                                       shift_range=3, read_heads=1, write_heads=1)

    # for layer in model.layers:
    #     print(layer.name, hasattr(layer, 'layer'))
    #     if hasattr(layer, 'layer'):
    #         print(layer.layer.name)
    # sys.exit(1)
    training_history = []
    evaluator = None
    keras.backend.get_session().run(tf.global_variables_initializer())
    # log_path = './logs'
    # tensorboard = TensorBoard(log_dir=log_path,
    #                           write_graph=True,  # This eats a lot of space. Enable with caution!
    #                           histogram_freq=1,
    #                           write_images=True,
    #                           batch_size=BATCH_SIZE,
    #                           write_grads=True,
    #                           embeddings_layer_names=['simple_ntm_1', 'simple_ntm_2'])

    # use a fraction of val data for quick validation
    val_data = val_data_q[0]
    if args.no_ntm:
        val_X, val_y = group_data(val_data, group_size, batch_size=None).next()
    else:
        val_X, val_y = group_data(val_data, group_size, batch_size=BATCH_SIZE).next()

    for epoch in range(args.epochs):
        train_data_q = train_input_gen.next()
        n_training_files = len(train_data_q)
        epoch_history = []
        start = time.time()
        for n, data in enumerate(train_data_q):
            if args.no_ntm:
                batch_generator = group_data(data, group_size, batch_size=None)
                fit_batch_size = group_size
            else:
                batch_generator = group_data(data, group_size, batch_size=BATCH_SIZE)
                fit_batch_size = BATCH_SIZE
            for X, y in batch_generator:
                if not y.any(): continue

                history = model.fit(X, y, batch_size=fit_batch_size, epochs=1, verbose=0, validation_split=0.0,
                                validation_data=[val_X, val_y], shuffle=False, class_weight=None, sample_weight=None,
                                initial_epoch=0)#, callbacks=[tensorboard])
                if history.history:
                    epoch_history.append(history.history)
            # reset states after a note file is processed
            model.reset_states()
            acc = np.mean([h['acc'] for h in epoch_history])
            loss = np.mean([h['loss'] for h in epoch_history])
            val_acc = np.mean([h['val_acc'] for h in epoch_history])
            val_loss = np.mean([h['val_loss'] for h in epoch_history])

            sys.stdout.write("epoch %d after training file %d/%d--- -%ds - loss : %.4f - acc : %.4f - val_loss : %.4f - val_acc : %.4f\r" % (
                epoch + 1, n + 1, n_training_files, int(time.time() - start), loss, acc, val_loss, val_acc))
            # sys.stdout.write("epoch %d after training file %d/%d--- -%ds - loss : %.4f - acc : %.4f\r" % (
            #     epoch + 1, n + 1, n_training_files, int(time.time() - start), loss, acc))
            sys.stdout.flush()

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

            if evaluator is None:
                evaluator = Evaluator(model, val_data_q)
            else:
                evaluator.model = model
            eval_results = evaluator.fast_eval()
            print(eval_results)

    print("Finished training... Performing evaluation...")
    evaluator = Evaluator(model, val_data_q)
    eval_results = evaluator.fast_eval()
    print(eval_results)
    with open(os.path.join(args.model_destination, 'results.pkl'), 'w') as f:
        pickle.dump(eval_results, f)
    with open(os.path.join(args.model_destination, 'history.json'), 'w') as f:
        json.dump(training_history, f)
    print("Done!")




if __name__ == "__main__":
    main()