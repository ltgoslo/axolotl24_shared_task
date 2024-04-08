import collections
import logging
import os.path
import pathlib

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from transformers import XGLMTokenizerFast, XGLMForCausalLM, AutoConfig


class Trainer:
    def __init__(self, args):
        self.args = args
        lang = os.path.split(args.test_data)[-1].split('.')[2]
        assert lang in {'fi', 'ru', 'surprise'}
        self.model_path = "pytorch_model.bin"
        self.mlp_path = "mlp.pt"
        self.model_dir = pathlib.Path(f"./{lang}_model")
        # we start with a simple CLM that we are going to adapt
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args.model_name)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.config.d_model,
                      self.config.d_model),
            nn.GELU(),
            nn.Linear(self.config.d_model,
                      self.config.d_model),
        )
        if args.path_to_checkpoint:
            state_dict = torch.load(
                    os.path.join(args.path_to_checkpoint, self.mlp_path),
                    map_location=args.device,
                )
            self.mlp.load_state_dict(state_dict)
        self.mlp.to(args.device)
        if args.path_to_checkpoint:
            self.model = XGLMForCausalLM.from_pretrained(
                args.path_to_checkpoint, config=self.config,
            ).to(args.device)
        else:
            self.model = XGLMForCausalLM.from_pretrained(args.model_name).to(args.device)
        self.tokenizer = XGLMTokenizerFast.from_pretrained(args.model_name)
        # we define an MLP to map context embeddings to model inputs

        self.optimizer = optim.AdamW(
            [*self.mlp.parameters(), *self.model.parameters()],
            weight_decay=0.001,
        )

    # helper function for logging purposes
    @staticmethod
    def safemean(q):
        if not len(q):
            return float("inf")
        return sum(q) / len(q)

    # helper function to retrieve the tokens corresponding to the headword in its token
    def make_headword_mask(self, inputs, indices, use_indices):
        mask = None
        index = torch.arange(inputs.input_ids.numel(),
                             device=self.args.device).unsqueeze(0)
        if not use_indices or pd.isna(indices):
            fst_tok, lst_tok = inputs.word_to_tokens(0)
            mask = ~((index >= fst_tok) & (index < lst_tok))
            mask = mask.unsqueeze(-1)
        else:
            for idx_span in indices.split(";"):
                fst_chr, lst_chr = map(int, idx_span.split(":"))
                fst_tok = inputs.char_to_token(fst_chr)
                lst_tok = inputs.char_to_token(lst_chr - 1)
                if fst_tok is None or lst_tok is None:
                    return None # this produces an error in outputs.masked_fill
                span_mask = ~((index >= fst_tok) & (index <= lst_tok))
                if mask is None:
                    mask = span_mask.unsqueeze(-1)
                else:
                    mask |= span_mask.unsqueeze(-1)
        return mask

    # helper function to remove unusable datapoints
    @staticmethod
    def remove_unusable(dataset, train=False):
        pre_drop = len(dataset)
        drop_subset = ['gloss', "example", "word"] if train else ["example", "word"]
        dataset = dataset.dropna(subset=drop_subset)
        post_drop = len(dataset)
        if pre_drop != post_drop:
            n_dropped = pre_drop - post_drop
            print(
                f"[Warn] some datapoints were dropped {n_dropped} / {pre_drop}, i.e. "
                f"{(n_dropped * 100) / pre_drop:.2f}%"
            )
        return dataset

    def do_train(
            self,
            dataset,
            epochs,
            use_indices=False,
    ):
        """train the model on the dataset for the provided number of epochs"""

        # we are going to make one prediction per word sense:
        # hence we start by grouping all contexts according to the wordsense they illustrate
        self.model.train()
        dataset = self.remove_unusable(dataset, train=True)
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

        # training loop.
        for epoch in tqdm.trange(epochs, desc="training"):
            with tqdm.trange(len(dataset), desc=f"train {epoch}") as pbar:
                dataset = dataset.sample(frac=1.0).reset_index(drop=True)
                xents = collections.deque(maxlen=512)

                def process(entry):
                    self.optimizer.zero_grad()
                    embeddings = []
                    for context, indices in zip(
                            entry["example"], entry["indices_target_token"]
                    ):
                        if not use_indices or pd.isna(indices):
                            context = entry["word"] + " " + context
                        inputs = self.tokenizer(
                            context, padding=True, truncation=True,
                            return_tensors="pt"
                        ).to(self.args.device)
                        outputs = self.model(**inputs, output_hidden_states=True)[
                            "hidden_states"
                        ][-1]
                        mask = self.make_headword_mask(
                            inputs, indices, use_indices,
                        )
                        headword_emb = outputs.masked_fill(mask, 0).sum(1)
                        context_emb = outputs.sum(1)
                        embeddings.append(
                            torch.cat([headword_emb, context_emb], dim=-1))
                    embeddings = torch.cat(embeddings, dim=0)

                    words = self.tokenizer(f"{entry['word']} : ",
                                      return_tensors="pt").to(
                        self.args.device
                    )
                    glosses = self.tokenizer(
                        f"{entry['gloss']} {self.tokenizer.eos_token}",
                        return_tensors="pt"
                    ).to(self.args.device)

                    # get input embeddings. You might need to tweak this depending on your exact model
                    inputs_embeds = [
                        self.mlp(embeddings).unsqueeze(0),
                        self.model.model.embed_tokens(words.input_ids),
                        self.model.model.embed_tokens(glosses.input_ids),
                    ]
                    inputs_embeds = torch.cat(inputs_embeds, dim=1)

                    # patch masks and labels
                    attention_mask = torch.ones(
                        inputs_embeds.shape[:2], dtype=torch.int,
                        device=self.args.device,
                    )
                    labels = torch.cat(
                        [
                            torch.full(
                                (1, inputs_embeds.size(
                                    1) - glosses.input_ids.numel()),
                                -100,
                                device=self.args.device,
                            ),
                            glosses.input_ids,
                        ],
                        dim=1,
                    )

                    # get loss
                    xent = self.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        labels=labels,
                        return_dict=True,
                    )["loss"]
                    xent.backward()
                    self.optimizer.step()

                    xents.append(xent.item())
                    pbar.set_postfix(
                        {
                            "xent": self.safemean(xents),
                        }
                    )
                    pbar.update()

                dataset.apply(process, axis=1)

        if not self.model_dir.exists():
            self.model_dir.mkdir()
        torch.save(self.mlp.state_dict(), self.model_dir / self.mlp_path)
        torch.save(self.model.state_dict(), self.model_dir / self.model_path)

    def do_test(self, dataset, use_indices=False):
        dataset = self.remove_unusable(dataset)
        old_senses = set(dataset[dataset.period == "old"].sense_id.unique())
        dataset = dataset[~dataset.sense_id.apply(old_senses.__contains__)]
        self.model.eval()
        self.mlp.eval()

        with torch.no_grad(), tqdm.trange(len(dataset),
                                          desc="embed contexts") as pbar:
            def embed(entry):
                contexts = entry["example"]
                indices = entry["indices_target_token"]
                if not use_indices or pd.isna(indices):
                    contexts = entry["word"] + " " + contexts
                inputs = self.tokenizer(
                    contexts, padding=True, truncation=True,
                    return_tensors="pt"
                ).to(self.args.device)
                outputs = \
                    self.model(**inputs, output_hidden_states=True)[
                        "hidden_states"][-1]
                mask = self.make_headword_mask(inputs, indices, use_indices)
                headword_emb = outputs.masked_fill(mask, 0).sum(1)
                context_emb = outputs.sum(1).detach()
                outputs = torch.cat([headword_emb, context_emb], dim=-1)
                pbar.update()
                return outputs.cpu()

            dataset["context embeddings"] = dataset.apply(embed, axis=1)
        dataset["sense_id"].fillna(0, inplace=True)
        dataset = (
            dataset[["sense_id", "word", "context embeddings"]]
            .groupby(["sense_id", "word"])["context embeddings"]
            .apply(list)
            .apply(lambda lst: torch.cat(lst, dim=0))
            .reset_index()
        )
        with tqdm.trange(len(dataset), desc="test") as pbar, torch.no_grad():
            def predict(entry):
                ctxt_embs = self.mlp(
                    entry["context embeddings"].to(self.args.device)
                )
                words = self.tokenizer(
                    f"{entry['word']} : ", return_tensors="pt").to(
                    self.args.device
                )
                # get input embeddings. You might need to tweak this depending on your exact model
                inputs_embeds = [
                    ctxt_embs.unsqueeze(0),
                    self.model.model.embed_tokens(words.input_ids),
                ]
                inputs_embeds = torch.cat(inputs_embeds, dim=1)

                attention_mask = torch.ones(*inputs_embeds.shape[:2],
                                            device=self.args.device)
                prompted = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                past_key_values = prompted["past_key_values"]
                predictions = prompted["logits"].argmax(-1)[:, -1].unsqueeze(1)
                out = self.model.generate(
                    input_ids=predictions,
                    attention_mask=torch.cat(
                        [attention_mask, torch.ones_like(predictions)], dim=1
                    ),
                    past_key_values=past_key_values,
                    do_sample=True,
                    max_new_tokens=100,
                )
                prediction = self.tokenizer.batch_decode(
                    out, skip_special_tokens=True
                )[0]
                pbar.update()
                return prediction

            dataset["predicted gloss"] = dataset.apply(predict, axis=1)

        dataset[["sense_id", "word", "predicted gloss"]].rename(
            columns={"predicted gloss": "gloss"}
        ).to_csv(self.args.predictions_file, sep="\t")
