import argparse
import sys

import numpy as np
import pandas as pd


def random_old_sense(pdf, seed):
    word2oldsenses = pdf[pdf.period == 'old'].groupby('word').sense_id.unique().to_dict()
    mask = (pdf.period == 'new')
    np.random.seed(seed)
    pdf.loc[mask, 'sense_id'] = pdf.loc[mask, 'word'].apply(lambda w: np.random.choice(word2oldsenses[w]))
    return pdf


def mfs_old_sense(df, seed):
    np.random.seed(seed)
    word2mfs = df[df.period=='old'].groupby('word').sense_id.apply(lambda r: np.random.choice(r.mode())).to_dict()
    pdf = df.copy()
    mask = (pdf.period=='new')
    pdf.loc[mask,'sense_id'] = pdf.loc[mask, 'word'].apply(lambda w: word2mfs[w])
    return pdf



def run_baseline(ftest, fpred, baseline_name='mfs_old_sense', seed=2024):
    """
    Randomly assign one of the old senses to each new usage.
    :param ftest: input file
    :param fpred: output file
    :param seed: random seed
    """
    df = pd.read_csv(ftest, sep='\t')
    baseline_fun = getattr(sys.modules[__name__], baseline_name)
    pdf = baseline_fun(df, seed)
    pdf.to_csv(fpred, sep="\t", index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--test", help="Path to the TSV file with the test data", required=True)
    arg("--pred", help="Path to the TSV file with system predictions", required=True)
    arg("--baseline_name", help="Baseline name: mfs_old_sense or random_old_sense", default="mfs_old_sense")
    arg("--seed", help="random seed", type=int, default=2024)
    return parser.parse_args()

def main():
    args = parse_args()
    run_baseline(args.test, args.pred, args.baseline_name, args.seed)

if __name__ == "__main__":
    main()

