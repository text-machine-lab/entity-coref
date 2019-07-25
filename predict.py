import argparse
import glob
import os
import pickle
import shutil
import subprocess
import re

from src.build_data import build_dataFrame, DataGen, group_data, slice_data
from src.evaluator import Evaluator, TriadEvaluator


def sort_files(file_list):
    # file_list.sort()
    order = 0
    file_hash = {}
    part_hash = {}

    for name in file_list:
        fname, part = name.split('-')
        if fname not in file_hash:
            file_hash[fname] = order
            order += 1

        part_n = re.search('\d+', part)
        if part_n:
            part_hash[name] = file_hash[fname] * 1000 + int(part_n.group(0))
        else:
            part_hash[name] = file_hash[fname] * 1000

    file_list.sort(key=lambda x: part_hash[x])

    return file_list


def scorer(path=None):
    if path is None:
        path = './scorers/v8.01/results/test/'
    if path[-1] != '/': path += '/'

    # keys = glob.glob(path+'keys/*')
    # keys = glob.glob('data/test/*conll')
    responses = sort_files(glob.glob(path+'responses/*'))
    # combined_key = path + 'key.tmp'
    combined_response = path + 'response.tmp'

    # with open(combined_key, 'wb') as f_key:
    #     for filename in keys:
    #         with open(filename, 'rb') as readfile:
    #             shutil.copyfileobj(readfile, f_key)

    with open(combined_response, 'wb') as f_response:
        for filename in responses:
            with open(filename, 'rb') as readfile:
                shutil.copyfileobj(readfile, f_response)

    cmd = 'perl scorers/v8.01/scorer.pl muc results/key.tmp %s none' %combined_response
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    print(proc.communicate()[0].decode('utf-8'))

    cmd = 'perl scorers/v8.01/scorer.pl bcub results/key.tmp %s none' %combined_response
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    print(proc.communicate()[0].decode('utf-8'))

    cmd = 'perl scorers/v8.01/scorer.pl ceafe results/key.tmp %s none' %combined_response
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    print(proc.communicate()[0].decode('utf-8'))

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("model_dir",
                        help="Directory containing the trained model")

    parser.add_argument("test_dir",
                        help="Directory containing test files")

    parser.add_argument("result_dir",
                        help="Directory to write result files")

    parser.add_argument("--max_distance",
                        default=15,
                        type=int,
                        help="Set the max entity distance to build data")

    parser.add_argument("--model_type",
                        default="gcl",
                        type=str,
                        help="chose 'keras', 'pytorch', or 'gcl' ")

    parser.add_argument("--clustering_only",
                        action='store_true',
                        default=False,
                        help="Use saved linkage files to perform clustering")

    parser.add_argument("--compute_linkage",
                        action='store_true',
                        default=False,
                        help="compute linkage, for clustering_only option")

    args = parser.parse_args()

    assert args.model_type in ('keras', 'pytorch', 'gcl')
    print(args.model_type)

    with open(os.path.join(args.model_dir, 'word_indexes.pkl'), 'rb') as f:
        word_indexes = pickle.load(f)
    with open(os.path.join(args.model_dir, 'pos_tags.pkl'), 'rb') as f:
        pos_tags = pickle.load(f)

    df = build_dataFrame(args.test_dir, threads=1, suffix='auto_conll')
    if args.clustering_only:
        model = None
    elif args.model_type == 'keras':
        from keras.models import load_model
        model = load_model(os.path.join(args.model_dir, 'model.h5'))
    # elif args.model_type == 'gcl':
    #     import torch
    #     from src.gcl_models import CorefTaggerGCL
    #     saved_model = torch.load(os.path.join(args.model_dir, 'model.pt'))
    #     model = CorefTaggerGCL(saved_model.vocab_size, saved_model.pos_size)
    #     model.load_state_dict(saved_model.state_dict())
    #     model.eval()
    else:
        import torch
        model = torch.load(os.path.join(args.model_dir, 'model.pt'))
        model.eval()
        print("model training mode", model.training)
    print("Loaded model")

    if args.model_type == 'gcl':
        n_files = len(df.doc_id.unique())
        test_gen = DataGen(df, word_indexes, pos_tags)
        test_input_gen = test_gen.generate_triad_input(looping=True, test_data=True, threads=4, max_distance=args.max_distance)
        # evaluator = TriadEvaluator(model, test_input_gen)
        group_size = 40 # 40
        evaluator = TriadEvaluator(model, test_input_gen, data_maker=group_data, group_size=group_size)

        # with torch.no_grad():
        evaluator.write_results(df, args.result_dir, n_iterations=n_files,
                                clustering_only=args.clustering_only, compute_linkage=args.compute_linkage)
    elif args.model_type == 'pytorch':
        n_files = len(df.doc_id.unique())
        test_gen = DataGen(df, word_indexes, pos_tags)
        test_input_gen = test_gen.generate_triad_input(looping=True, test_data=True, threads=4,
                                                       max_distance=args.max_distance)
        # evaluator = TriadEvaluator(model, test_input_gen)
        group_size = 100
        evaluator = TriadEvaluator(model, test_input_gen, data_maker=slice_data, group_size=group_size)

        evaluator.write_results(df, args.result_dir, n_iterations=n_files,
                                clustering_only=args.clustering_only, compute_linkage=args.compute_linkage)
    else:
        test_gen = DataGen(df, word_indexes, pos_tags)
        test_data_q = next(test_gen.generate_input(looping=False, test_data=True))
        print("Loaded test data")

        evaluator = Evaluator(model, test_data_q)
        print("Performing fast evaluation...")
        print(evaluator.fast_eval())
        print("Saving result files...")
        evaluator.write_results(df, args.result_dir)

    scorer(args.result_dir)

if __name__ == "__main__":
    main()
