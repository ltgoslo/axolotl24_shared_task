import argparse
import csv
from glob import glob
import os

import pandas as pd

ID_SENSE = "identifier_sense"
ID = "identifier"


def read_csv(path):
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dwug_path', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dwug_path = args.dwug_path
    test = pd.DataFrame()
    for word_path in glob(f"{dwug_path}data/*"):
        word = os.path.split(word_path)[-1]

        word_uses = read_csv(os.path.join(dwug_path, "data", word, "uses.csv"))
        word_senses = read_csv(
            os.path.join(dwug_path, "data", word, "senses.csv")
        )
        judgements = read_csv(
            os.path.join(dwug_path, "data", word, "judgments_senses.csv")
        )
        judgements.dropna(subset=ID_SENSE, inplace=True)
        judgements_majory_voting = pd.DataFrame()
        for sample in judgements[ID].unique():
            this_sample = judgements[judgements[ID] == sample]
            most_often = this_sample[ID_SENSE].mode().iloc[0]
            to_concat = this_sample[
                this_sample[ID_SENSE] == most_often
                ].reset_index().iloc[0]
            judgements_majory_voting = pd.concat(
                [judgements_majory_voting, to_concat], axis=1,
            )
        judgements_majory_voting = judgements_majory_voting.T
        judgements_majory_voting = judgements_majory_voting.set_index(
            "index"
        ).reset_index()
        senses_glossed = word_senses.join(
            judgements_majory_voting.set_index(ID_SENSE), on=ID_SENSE,
        )
        this_word = word_uses.join(
            senses_glossed.set_index(ID), on=ID, lsuffix="_uses",
        )
        if len(
                this_word[
                    ~this_word.description_sense.isna()
                ].grouping.unique()
        ) < 2:
            continue
        else:
            nans = this_word[this_word.description_sense.isna()].index
            if len(nans) > 0:
                this_word.drop(nans, inplace=True)

        if len(
                this_word[
                    this_word.description_sense != 'andere'
                ].grouping.unique()
        ) < 2:
            continue
        else:
            andere = this_word[this_word.description_sense == 'andere'].index
            if len(andere) > 0:
                this_word.drop(andere, inplace=True)

        gr = {1: "old", 2: "new"}
        test = pd.concat(
            [
                test,
                pd.DataFrame(
                    {
                        "usage_id": this_word[ID],
                        "word": this_word.lemma,
                        "orth": this_word.lemma,
                        "sense_id": this_word[ID_SENSE].apply(
                            lambda x: f"{word}_{x}"
                        ),
                        "gloss": this_word["description_sense"],
                        "example": this_word.context,
                        "indices_target_token": this_word.indexes_target_token,
                        "date": this_word.date,
                        "period": this_word.grouping.apply(lambda x: gr[x]),
                    },
                ),
            ],
        )
    to_shuffle = []
    with open('targets.txt', 'r', encoding='utf8') as words:
        for word in words:
            this_word = test[test.word == word.rstrip()]
            this_word = this_word.sample(frac=1, random_state=42)
            to_shuffle.append(this_word)
    new_df = pd.concat(to_shuffle)
    assert new_df.shape == test.shape
    new_df["usage_id"] = [f"test_surprise_{i}" for i in range(new_df.shape[0])]
    new_df.to_csv("axolotl.test.surprise.gold.tsv", index=False, sep="\t")
    print(new_df.shape)
    print(f"NaNs: {new_df[new_df.gloss.isna()].shape[0]}")
    print(f"Andere: {new_df[new_df.gloss == 'andere'].shape[0]}")


if __name__ == '__main__':
    main()
