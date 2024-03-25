#!/usr/bin/env python3
# coding: utf-8

import argparse
import itertools
import evaluate
import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU
import torch

# Scoring script to evaluate submissions to Track 1 of the AXOLOTL'24 shared task
# https://github.com/ltgoslo/axolotl24_shared_task/

p = argparse.ArgumentParser()
p.add_argument("submission")
p.add_argument("reference")
p.add_argument("output")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sub = pd.read_csv(args.submission, sep="\t")
sub = sub.fillna("")
ref = pd.read_csv(args.reference, sep="\t")
old_senses = set(ref[ref.period == "old"].sense_id.unique())
ref = ref[~ref.sense_id.apply(old_senses.__contains__)].reset_index()

sub = sub[~sub.sense_id.apply(old_senses.__contains__)].reset_index()  # sanity check
assert len(sub) != 0, "no (usable) predictions"
verbose = True
apply_iou_penalty = False
normalize_penalty = False

words_sub = set(sub.word.unique())
words_ref = set(ref.word.unique())
if len(words_sub & words_ref) == 0:
    scores = {
        "bleu": 0,
        "bertscore": 0,
        "IoU": 0,
        "delta": ref.groupby("word").size().mean(),
    }
else:
    # wrongfully included / omitted words penalty
    iou = len(words_sub & words_ref) / len(words_sub | words_ref)
    bertscorer = evaluate.load("bertscore")
    bleuscorer = BLEU(effective_order=True, smooth_method='exp')  # default smoothing, but I'm being explicit
    words = words_sub & words_ref
    records = []
    for word in words:
        pred_novel_senses = sub[sub.word == word].gloss.to_list()
        n_pred = len(pred_novel_senses)
        true_novel_senses = ref[ref.word == word].gloss.unique().tolist()
        n_true = len(true_novel_senses)
        sense_preds, sense_refs = zip(
            *itertools.product(pred_novel_senses, true_novel_senses)
        )
        all_bert_scores = np.array(
            bertscorer.compute(
                predictions=sense_preds,
                references=sense_refs,
                model_type="bert-base-multilingual-cased",
                device=device,
            )["f1"]
        ).reshape(n_pred, n_true)
        alignments = []
        summed_bertscores = 0.0
        for runs in range(min(n_pred, n_true)):
            i, j = np.unravel_index(all_bert_scores.argmax(), all_bert_scores.shape)
            alignments.append((i, j))
            summed_bertscores += all_bert_scores[i, j]
            all_bert_scores[i, :] = -float("inf")
            all_bert_scores[:, j] = -float("inf")
        summed_bleuscores = 0.0
        for i, j in alignments:
            summed_bleuscores += bleuscorer.sentence_score(
                pred_novel_senses[i], [true_novel_senses[j]]
            ).score
        norm = max(n_true, n_pred) if normalize_penalty else len(alignments)
        records.append(
            {
                "bleu": (summed_bleuscores / 100) / norm,
                "bertscore": summed_bertscores / norm,
                "IoU": iou,
                "delta": abs(n_true - n_pred),
            }
        )

    scores_df = pd.DataFrame.from_records(records)
    scores = {col: scores_df[col].mean() for col in scores_df.columns}
    if apply_iou_penalty:
        scores["bleu"] *= iou
        scores["bertscore"] *= iou

if verbose:
    import pprint

    pprint.pprint(scores)

with open(args.output, "w") as ostr:
    print(f"BLEU: {scores['bleu']:0.3f}", file=ostr)
    print(f"BERTScore: {scores['bertscore']:0.3f}", file=ostr)
