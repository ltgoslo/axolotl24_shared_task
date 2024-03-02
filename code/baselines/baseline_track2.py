import argparse
import collections
import pathlib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type=pathlib.Path, help="path to training data")
parser.add_argument(
    "--test_data",
    type=pathlib.Path,
    help="path to test data.",
)
parser.add_argument(
    "--predictions_file",
    type=pathlib.Path,
    help="path to prediction output file.",
    default="preds.tsv",
)
parser.add_argument(
    "--model_name", default="TurkuNLP/gpt3-finnish-small", help="base model"
)
parser.add_argument(
    "--device",
    default=(
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    ),
    type=torch.device,
    help="device to train the model on.",
)
args = parser.parse_args()

dataset = pd.read_csv(args.train_data, sep="\t")

# we start with a simple CLM that we are going to adapt
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)
model.train()

# we are going to make one prediction per word sense:
# hence we start by grouping all contexts according to the wordsense they illustrate
dataset_ = (
    dataset[["sense_id", "word", "gloss", "example"]]
    .groupby(["sense_id", "gloss", "word"])["example"]
    .apply(list)
    .reset_index()
)
target_indices = (
    dataset[["sense_id", "word", "gloss", "indices_target_token"]]
    .groupby(["sense_id", "gloss", "word"])["indices_target_token"]
    .apply(list)
    .reset_index()
)["indices_target_token"]
dataset_["indices_target_token"] = target_indices
dataset = dataset_

# we define an MLP to map context embeddings to model inputs
mlp = nn.Sequential(
    nn.Linear(2 * model.config.hidden_size, model.config.hidden_size),
    nn.GELU(),
    nn.Linear(model.config.hidden_size, model.config.hidden_size),
).to(args.device)

optimizer = optim.AdamW([*mlp.parameters(), *model.parameters()], weight_decay=0.001)


def safemean(q):
    if not len(q):
        return float("inf")
    return sum(q) / len(q)


# helper function to retrieve the tokens corresponding to the headword in its token
def make_headword_mask(inputs, indices):
    mask = None
    index = torch.arange(
        inputs.input_ids.numel(), device=args.device
    ).unsqueeze(0)
    for idx_span in indices.split(";"):
        fst_chr, lst_chr = map(int, idx_span.split(":"))
        fst_tok = inputs.char_to_token(fst_chr)
        lst_tok = inputs.char_to_token(lst_chr - 1)
        if fst_tok is None or lst_tok is None:
            return None
        span_mask = ~((index >= fst_tok) & (index <= lst_tok))
        if mask is None:
            mask = span_mask.unsqueeze(-1)
        else:
            mask |= span_mask.unsqueeze(-1)
    return mask


# training loop.
for epoch in tqdm.trange(5, desc="training"):
    with tqdm.trange(len(dataset), desc=f"train {epoch}") as pbar:
        dataset = dataset.sample(frac=1.0).reset_index(drop=True)
        xents = collections.deque(maxlen=512)

        def process(entry):
            optimizer.zero_grad()
            embeddings = []
            for context, indices in zip(
                entry["example"], entry["indices_target_token"]
            ):
                inputs = tokenizer(
                    context, padding=True, truncation=True, return_tensors="pt"
                ).to(args.device)
                outputs = model(**inputs, output_hidden_states=True)["hidden_states"][
                    -1
                ]
                mask = make_headword_mask(inputs, indices)
                headword_emb = outputs.masked_fill(mask, 0).sum(1)
                context_emb = outputs.sum(1)
                embeddings.append(torch.cat([headword_emb, context_emb], dim=-1))
            embeddings = torch.cat(embeddings, dim=0)

            words = tokenizer(f"{entry['word']} : ", return_tensors="pt").to(
                args.device
            )
            glosses = tokenizer(
                f"{entry['gloss']} {tokenizer.eos_token}", return_tensors="pt"
            ).to(args.device)

            # get input embeddings. You might need to tweak this depending on your exact model
            inputs_embeds = [
                mlp(embeddings).unsqueeze(0),
                model.transformer.word_embeddings(words.input_ids),
                model.transformer.word_embeddings(glosses.input_ids),
            ]
            inputs_embeds = torch.cat(inputs_embeds, dim=1)

            # patch masks and labels
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.int, device=args.device
            )
            labels = torch.cat(
                [
                    torch.full(
                        (1, inputs_embeds.size(1) - glosses.input_ids.numel()),
                        -100,
                        device=args.device,
                    ),
                    glosses.input_ids,
                ],
                dim=1,
            )

            # get loss
            xent = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )["loss"]
            xent.backward()
            optimizer.step()

            xents.append(xent.item())
            pbar.set_postfix(
                {
                    "xent": safemean(xents),
                }
            )
            pbar.update()

        dataset.apply(process, axis=1)

modeldir = pathlib.Path(".")
torch.save(mlp.state_dict(), modeldir / f"mlp.pt")
torch.save(model.state_dict(), modeldir / f"model.pt")

test_dataset = pd.read_csv(args.test_data, sep="\t")
old_senses = set(test_dataset[test_dataset.period == "old"].sense_id.unique())
test_dataset = test_dataset[~test_dataset.sense_id.apply(old_senses.__contains__)]
model.eval()
mlp.eval()

with torch.no_grad(), tqdm.trange(len(test_dataset), desc="embed contexts") as pbar:

    def embed(entry):
        contexts = entry["example"]
        inputs = tokenizer(
            contexts, padding=True, truncation=True, return_tensors="pt"
        ).to(args.device)
        outputs = model(**inputs, output_hidden_states=True)["hidden_states"][-1]
        mask = make_headword_mask(inputs, entry["indices_target_token"])                
        headword_emb = outputs.masked_fill(mask, 0).sum(1).detach()
        context_emb = outputs.sum(1).detach()
        outputs = torch.cat([headword_emb, context_emb], dim=-1)
        pbar.update()
        return outputs.cpu()

    test_dataset["context embeddings"] = test_dataset.apply(embed, axis=1)

test_dataset = (
    test_dataset[["sense_id", "word", "gloss", "context embeddings"]]
    .groupby(["sense_id", "gloss", "word"])["context embeddings"]
    .apply(list)
    .apply(lambda lst: torch.cat(lst, dim=0))
    .reset_index()
)

with tqdm.trange(len(test_dataset), desc="test") as pbar, torch.no_grad():

    def predict(entry):
        ctxt_embs = mlp(entry["context embeddings"].to(args.device))
        words = tokenizer(f"{entry['word']} : ", return_tensors="pt").to(args.device)
        # get input embeddings. You might need to tweak this depending on your exact model
        inputs_embeds = [
            ctxt_embs.unsqueeze(0),
            model.transformer.word_embeddings(words.input_ids),
        ]
        inputs_embeds = torch.cat(inputs_embeds, dim=1)

        attention_mask = torch.ones(*inputs_embeds.shape[:2], device=args.device)
        prompted = model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True
        )
        past_key_values = prompted["past_key_values"]
        predictions = prompted["logits"].argmax(-1)[:, -1].unsqueeze(1)
        out = model.generate(
            input_ids=predictions,
            attention_mask=torch.cat(
                [attention_mask, torch.ones_like(predictions)], dim=1
            ),
            past_key_values=past_key_values,
            do_sample=True,
            max_new_tokens=100,
        )
        prediction = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        pbar.update()
        return prediction

    test_dataset["predicted gloss"] = test_dataset.apply(predict, axis=1)

test_dataset[["sense_id", "word", "predicted gloss"]].rename(
    columns={"predicted gloss": "gloss"}
).to_csv(args.predictions_file, sep="\t")
