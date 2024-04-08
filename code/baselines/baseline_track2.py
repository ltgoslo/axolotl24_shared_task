import argparse
import pathlib
import pandas as pd
import torch

from baseline_track_2.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data", type=pathlib.Path, help="path to training data",
    )
    parser.add_argument(
        "--pretrain_data", type=pathlib.Path, nargs="*",
        help="path to training data"
    )
    parser.add_argument(
        "--test_data",
        type=pathlib.Path,
        help="path to test data.",
        required=True,
    )
    parser.add_argument(
        "--predictions_file",
        type=pathlib.Path,
        help="path to prediction output file.",
        default="preds.tsv",
    )
    parser.add_argument("--model_name", default="facebook/xglm-564M",
                        help="base model")
    parser.add_argument(
        "--pretrain_epochs",
        default=1,
        type=int,
        help="number of iterations over training data",
    )
    parser.add_argument(
        "--epochs", default=1, type=int,
        help="number of iterations over training data"
    )
    parser.add_argument(
        "--device",
        default=(
            torch.device(
                "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
        type=torch.device,
        help="device to train the model on.",
    )
    parser.add_argument(
        '--path_to_checkpoint',
        default='',
    )
    parser.add_argument(
        '--debug_samples',
        default=0,
        type=int,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(args)
    if args.train_data is not None:
        pretrain_dataset = (
            pd.concat(
                [pd.read_csv(pretrain_file, sep="\t") for pretrain_file in
                 args.pretrain_data]
            )
            if args.pretrain_data
            else None
        )
        train_dataset = pd.read_csv(args.train_data, sep="\t")
        if args.debug_samples:
            train_dataset = train_dataset.iloc[:args.debug_samples]
        global_use_indices = not train_dataset["indices_target_token"].isna().any()
        print(f"relying on indices in train? {global_use_indices}")
        if pretrain_dataset is not None:
            trainer.do_train(
                pretrain_dataset,
                args.pretrain_epochs,
                use_indices=global_use_indices,
            )
        trainer.do_train(
            train_dataset,
            args.epochs,
            use_indices=global_use_indices,
        )
    test_dataset = pd.read_csv(args.test_data, sep="\t")
    if args.debug_samples:
        test_dataset = test_dataset.iloc[:args.debug_samples]
    #global_use_indices = not test_dataset["indices_target_token"].isna().any()
    # see the bug in Trainer.make_headword_mask
    global_use_indices = False

    print(f"relying on indices in test? {global_use_indices}")
    trainer.do_test(
        test_dataset,
        use_indices=global_use_indices,
    )


if __name__ == '__main__':
    main()
