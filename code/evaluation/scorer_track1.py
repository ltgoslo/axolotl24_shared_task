#!/usr/bin/env python3
# coding: utf-8

import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, f1_score
from tqdm import tqdm

# Scoring script to evaluate submissions to Track 1 of the AXOLOTL'24 shared task
# https://github.com/ltgoslo/axolotl24_shared_task/

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--gold", "-g", help="Path to the TSV file with gold data", required=True)
    arg(
        "--pred",
        "-p",
        help="Path to the TSV file with system predictions",
        required=True,
    )
    arg("--output", "-o", help="Path to the output file", default="track1_out.txt")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    gold_data = pd.read_csv(args.gold, sep="\t")
    predictions = pd.read_csv(args.pred, sep="\t")

    assert len(gold_data) == len(predictions)
    assert (
            gold_data[gold_data.period == "new"].example.tolist()
            == predictions[predictions.period == "new"].example.tolist()
    )

    logger.info(f"Data loaded from {args.gold} and {args.pred}")
    logger.info(f"{len(gold_data)} example usages")
    logger.info("Computing Adjusted Rand Index and F1 for predicted senses...")

    ari_scores = []
    f1_scores = []

    for targetword in tqdm(gold_data.word.unique()):
        gold_senses = gold_data[
            (gold_data.word == targetword) & (gold_data.period == "new")
            ].sense_id.values
        pred_senses = predictions[
            (predictions.word == targetword) & (predictions.period == "new")
            ].sense_id.values
        ari = adjusted_rand_score(gold_senses, pred_senses)
        ari_scores.append(ari)
        logger.debug(f"ARI for {targetword}: {ari}")
        old_senses = set(
            gold_data[(gold_data.word == targetword) & (gold_data.period == "old")]
                .sense_id.unique()
                .tolist()
        )
        if len(old_senses) == 0:
            logger.info(f"Not computing F1 for {targetword}: no old senses")
            continue
        test_usages = gold_data[
            (gold_data.word == targetword)
            & (gold_data.period == "new")
            & (gold_data.sense_id.isin(old_senses))
            ]
        test_usages_ids = set(test_usages.usage_id.tolist())
        if len(test_usages) == 0:
            test_usages_predicted = predictions[
                (predictions.word == targetword)
                & (predictions.period == "new")
                & (predictions.sense_id.isin(old_senses))
                ]
            if len(test_usages_predicted) == 0:
                f1_scores.append(1.0)
                logger.info(
                    f"Macro F1 set to 1.0 for {targetword}: "
                    f"no new usages with old senses, and none predicted"
                )
            else:
                f1_scores.append(0.0)
                logger.info(
                    f"Macro F1 set to 0.0 for {targetword}: "
                    f"old senses predicted when there are none"
                )
            continue
        test_usages_gold_senses = test_usages.sense_id.tolist()
        test_usages_predictions = predictions[predictions.usage_id.isin(test_usages_ids)]
        test_usages_predicted_senses = test_usages_predictions.sense_id.tolist()
        assert len(test_usages_gold_senses) == len(test_usages_predicted_senses)
        test_usages_predicted_senses = [
            "novel" if el not in old_senses else el
            for el in test_usages_predicted_senses
        ]
        f1 = f1_score(
            test_usages_gold_senses,
            test_usages_predicted_senses,
            average="macro",
            zero_division=0.0,
        )
        logger.debug(f"Macro F1 for {targetword}: {f1}")
        f1_scores.append(f1)

    average_ari = np.mean(ari_scores)
    logger.info(
        f"Average ARI across {len(ari_scores)} target words: {average_ari:0.3f}"
    )
    average_f1 = np.mean(f1_scores)
    logger.info(
        f"Average macro-F1 across {len(f1_scores)} target words: {average_f1:0.3f}"
    )

    with open(args.output, "w") as out:
        print(f"ARI: {average_ari:0.3f}", file=out)
        print(f"F1: {average_f1:0.3f}", file=out)
