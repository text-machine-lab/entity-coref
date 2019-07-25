"""Train the triad model"""
import argparse
import os
import pickle

from src.build_data import build_dataFrame, DataGen


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("train_dir",
                        help="Directory containing training annotations")

    parser.add_argument("model_destination",
                        help="Where to store the trained model")

    parser.add_argument("--val_dir",
                        default=None,
                        help="Directory containing validation annotations")

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

    parser.add_argument("--model_type",
                        default="gcl",
                        type=str,
                        help="chose 'keras', 'pytorch', or 'gcl' ")

    args = parser.parse_args()

    assert os.path.isdir(args.train_dir)
    assert os.path.isdir(args.model_destination)

    train_gen = DataGen(build_dataFrame(args.train_dir, threads=3))
    with open(os.path.join(args.model_destination, 'word_indexes.pkl'), 'wb') as f:
        pickle.dump(train_gen.word_indexes, f)
    with open(os.path.join(args.model_destination, 'pos_tags.pkl'), 'wb') as f:
        pickle.dump(train_gen.pos_tags, f)

    assert args.model_type in ('keras', 'pytorch', 'gcl')
    if args.model_type == 'keras':  # keras model
        from src.keras_models import train
        train(train_gen=train_gen,
              model_destination=args.model_destination,
              val_dir=args.val_dir,
              load_model=args.load_model,
              epochs=args.epochs)
    elif args.model_type == 'pytorch':  # pytorch model
        from src.torch_models import train
        train(train_gen=train_gen,
              model_destination=args.model_destination,
              val_dir=args.val_dir,
              load_model=args.load_model,
              epochs=args.epochs)
    else:  # pytorch gcl model
        from src.gcl_models import train
        train(train_gen=train_gen,
              model_destination=args.model_destination,
              val_dir=args.val_dir,
              load_model=args.load_model,
              epochs=args.epochs,
              group_size=40,
              batch_size=25,
              model_type=args.model_type)

if __name__ == "__main__":
    main()