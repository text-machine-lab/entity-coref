import argparse
import glob
import os
import pickle
import shutil

from src.build_data import build_dataFrame, DataGen, group_data, slice_data
from src.evaluator import Evaluator, TriadEvaluator


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

    parser.add_argument("--keras",
                        action='store_true',
                        default=False,
                        help="Use keras model")

    args = parser.parse_args()


    if not args.score_only:
        with open(os.path.join(args.model_dir, 'word_indexes.pkl'), 'rb') as f:
            word_indexes = pickle.load(f)
        with open(os.path.join(args.model_dir, 'pos_tags.pkl'), 'rb') as f:
            pos_tags = pickle.load(f)

        df = build_dataFrame(args.test_dir, threads=1, suffix='auto_conll')
        if args.keras:
            from keras.models import load_model
            model = load_model(os.path.join(args.model_dir, 'model.h5'))
        else:
            import torch
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
    main()
