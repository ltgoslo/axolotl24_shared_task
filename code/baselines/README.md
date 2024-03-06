# Track 1

## Example usage

### Running a baseline for track 1

```commandline
cd scripts/baselines/
python3 baseline_track1.py --test ../../data/axolotl.ru/axolotl.dev.ru.tsv --pred pred_dev_ru.tsv
```

### Evaluating results for track 1

```commandline
cd ../evaluation/
python3 scorer_track1.py --gold ../../data/axolotl.ru/axolotl.dev.ru.tsv --pred ../baselines/pred_dev_ru.tsv
```

### Baseline results for track 1

| Language     | ARI   | MACRO F1 |
|--------------|-------|----------|
| Finnish, dev | 0.022 | 0.222    |
| Russian, dev | 0.073 | 0.280    |

## How this baseline works

1. Get sentence embedding representations for the "new" (1st time period) usage examples
and "old" (2nd time period) senses using a BERT-like
language model.

2. Cluster the "new" usage examples using their sentence representations as features.

3. Loop over all old senses and calculate cosine similarity between its embedding
and the embedding of the first example usage for each cluster (both embeddings are those that were generated at the step 1). If similarity is above a
predefined threshold, assign the current sense to this cluster and move on to the next sense.

4. If a cluster was not assigned any sense after looping over all "old" senses,
this cluster forms a novel sense.

This baseline may be improved in many ways. e.g. another sentence embedding model may be chosen, 
"old" example usages may be added to the "old" senses (but remember that many Russian "old" senses lack usage examples),
clustering may be done better, the best cluster embedding may be not that of the first usage example in it etc.

Feel free also to abandon this baseline at all and use methods other than calculating similarity between transformer-based sentence embeddings.


# Track 2


## Example usage

### Running a baseline for track 2

```commandline
cd scripts/baselines/
python3 baseline_track2.py --train_data ../../data/axolotl.fi/axolotl.train.fi.tsv --test_data ../../data/axolotl.fi/axolotl.dev.fi.tsv --predictions_file ../baselines/pred_dev_fi.track2.tsv

```

<!-- ### Evaluating results for track 2

```commandline
cd ../evaluation/
python3 scorer_track2.py --gold ../../data/axolotl.fi/axolotl.dev.fi.tsv --pred ../baselines/pred_dev_fi.track2.tsv
```

-->

### Baseline results for track 2

TBA


| Language     | BLEU | BERTScore |
|--------------|------|-----------|
| Finnish, dev |      |           |
| Russian, dev |      |           |

## How this baseline works

**NB:** This baseline presupposes that you have added a sense ID prediction to the test split (i.e., that you have tried to cluster senses before attempting to explain them). See below for suggestions on how to do without.

The baseline works by fine-tuning a large generative language model as follows:

1. For every data point, get a context and headword embedding representations using this model;

2. For every provided sense ID, group all relevant context and headword embeddings as inputs and use them prompt the model to generate the definition of a given sense.

In other words, we do two sets of pass for each sense: the first set in step 1. allows us to retrieve input features, the second is meant to generate the actual definition.

This baseline is highly speculative and may or may not be a good idea. Feel free to improve on it in any way you see fit. You could for instance:
- rely on a separate model for encoding context and headword embeddings;
- consider using a sequence-to-sequence encoder-decoder architecture (e.g. mBART) instead of the GPT-like model we used;
- abandon this baseline altogether and devlop a completely different approach.

Also keep in mind that our baseline system assumes predictions for track 1. We believe this is the most natural way to approach the task: first identify which words need updated senses, and then predict what these updated senses should mean.

You are however more than welcome to explore alternative approaches! You could for instance try to generate all definitions at once. You could try to guess how polysemous the target word is without explicitly clustering these senses (e.g., by looking at the variability of its embeddings in context) and use some form of Bayesian or GAN-like approach to generate multiple distinct predictions.
