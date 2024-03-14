import argparse
import logging
import random
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from sklearn.cluster import AffinityPropagation
import numpy as np
from tqdm import tqdm

NEW_PERIOD = "new"
OLD_PERIOD = "old"
SENSE_ID_COLUMN = "sense_id"
USAGE_ID_COLUMN = "usage_id"
PERIOD_COLUMN = "period"

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--test", help="Path to the TSV file with the test data", required=True)
    arg("--pred", help="Path to the TSV file with system predictions", required=True)
    arg("--model", help="Sentence embedding model", default="setu4993/LEALLA-large")
    arg("--st", help="Similarity threshold", type=float, default=0.3)
    arg(
        "--device",
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
        type=torch.device,
        help="Device to load the model on.",
    )
    return parser.parse_args()


def load_model(arguments):
    logging.info(f"Loading model {arguments.model} for sentence embeddings")
    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    model = AutoModel.from_pretrained(arguments.model).to(arguments.device)
    model = model.eval()
    logging.info(f"Loaded model {arguments.model}")
    return tokenizer, model


def main():
    args = parse_args()
    tokenizer, model = load_model(args)
    targets = pd.read_csv(args.test, sep="\t")
    for target_word in tqdm(targets.word.unique()):
        this_word = targets[targets.word == target_word]
        new = this_word[this_word[PERIOD_COLUMN] == NEW_PERIOD]
        old = this_word[this_word[PERIOD_COLUMN] == OLD_PERIOD]
        new_examples = new.example.to_list()
        new_usage_ids = new[USAGE_ID_COLUMN]
        old_glosses = [
            f"{gl} {ex}".strip() if isinstance(ex, str) else gl
            for gl, ex in zip(old.gloss.to_list(), old.example.to_list())
        ]
        senses_old = old[SENSE_ID_COLUMN].to_list()
        latin_name = senses_old[0].split("_")[0]

        # Getting representations for the new examples and old senses

        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": 256,
        }
        new_inputs = tokenizer(new_examples, **tokenizer_kwargs).to(args.device)
        old_inputs = tokenizer(old_glosses, **tokenizer_kwargs).to(args.device)
        with torch.no_grad():
            new_outputs = model(**new_inputs).pooler_output
            old_outputs = model(**old_inputs).pooler_output

        # Clustering the new representations in order to get new senses
        old_outputs = old_outputs.cpu()
        ap = AffinityPropagation(random_state=42)
        new_numpy = new_outputs.detach().cpu().numpy()
        clustering = ap.fit(new_numpy)

        # Aligning the old and new senses

        exs2senses = {}
        seen = set()
        for label in np.unique(clustering.labels_):
            found = ""
            examples_indices = np.where(clustering.labels_ == label)[0]
            examples = [new_examples[i] for i in examples_indices]
            this_cluster = new_numpy[clustering.labels_ == label]
            emb1 = torch.Tensor(this_cluster[0])
            for emb2, defs, sense_old in zip(old_outputs, old_glosses, senses_old):
                if sense_old not in seen:
                    sim = F.cosine_similarity(emb1, emb2, dim=0)
                    if sim.item() >= args.st:
                        found = sense_old
                        seen.add(sense_old)
                        break
            if not found:
                found = f"{latin_name}_novel_{label}"
            for ex in examples:
                exs2senses[ex] = found

        assert len(new_examples) == new_usage_ids.shape[0]
        for usage_id, example in zip(new_usage_ids, new_examples):
            system_answer = exs2senses[example]
            row_number = targets[targets[USAGE_ID_COLUMN] == usage_id].index
            targets.loc[row_number, SENSE_ID_COLUMN] = system_answer
    logging.info(f"Writing the result to {args.pred}")
    targets.to_csv(args.pred, sep="\t", index=False)


if __name__ == "__main__":
    main()
