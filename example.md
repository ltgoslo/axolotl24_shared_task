# AXOLOTL in one example

Welcome to the AXOLOTL-24 Shared Task on Explainable Semantic Change Modeling! 
"AXOLOTL" stands for "Ascertain and eXplain Overhauls of the Lexicon Over Time at LChange'24".

Below, we explain the essence of two AXOLOTL subtasks in a small and comprehensible English example.

## Subtask 1
In the first subtask, you are given a set of target words, each accompanied by two sets of usages (a text fragment containing the word): from the older time period (**A**) and the more modern time period (**B**). 
In addition, a dictionary’s **sense inventory** with sense IDs for tim period **A** is given.

For instance, for the English target word "**cell**" in **time period A**, a sense inventory and a set of usage examples for each sense are given:

1. `cell_1`: _A subordinate monastic establishment_ 
   - "In Ireland there are innumerable chapels and _cells_ and more than 400 churches" 
   - "Edward the Confessor established a _cell_ of monks on the site of his old wooden church." 
2. `cell_2`: _An individual cell for a monk or nun in a monastic community_ 
   - "The _cells_ of the Celtic monks are preserved and cleaned"
   - "In a Convent there are numerous little _cells_." 

Fom the **time period B** only a set of usages is provided:

1. "The monks are staying in their _cells_."
2. "Gregor Mendel spent a good amount of time outside his _cell_."
3. "An American company has applied to experiment in Britain on Parkinson's disease sufferers by injecting their brains with _cells_ from pigs."
4. "In multicellular organisms, groups of _cells_ form tissues and tissues come together to form organs."

Comparing the usages from the two time periods, we can notice that the _third_ and the _fourth_ usages in the time period **B**  correspond to a newly gained biological sense, which is absent in time period **A**. This sense is not covered by the provided sense inventory, therefore a system should assign these usages a new sense ID (for example, `cell_3`), while re-using the existing sense IDs for the rest of the usages:

| **Usages from time period B**                                                                                                                | **sense_id** |
|----------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| "The monks are staying in their _cells_"                                                                                                     | `cell_2`     |
| "Gregor Mendel spent a good amount of time outside his _cell_"                                                                               | `cell_2`     |
| "An American company has applied to experiment in Britain on Parkinson's disease sufferers by injecting their brains with _cells_ from pigs" | `cell_3`     |
| "In multicellular organisms, groups of _cells_ form tissues and tissues come together to form organs"                                        | `cell_3`     |

The performance of the system is evaluated by comparing its predictions with the gold data using Adjusted Rand Index (ARI) for all the usages from time period **B** and macro F1 score for all **B** usages with senses already existing in **A**.

See [our evaluation scripts](https://github.com/ltgoslo/axolotl24_shared_task/tree/main/code/evaluation) for more details.

## Subtask 2
The second subtask involves the creation of **dictionary-like definitions** for the **novel (gained) senses** of every target word.
The definitions can be based on existing ontologies or generated from scratch: this is completely up to the participants. 

If a participant came up with a solution for the Subtask 1, novel senses are inferred trivially from the Subtask 1 predictions (all the senses of the usages in time period **B** not attested in the **A** sense inventory). However, it is also possible to solve Subtask 2 independently, without Subtask 1. In the example below, we continue with the English word "cell", assuming that the participant already has a hypothetical solution for Subtask 1.

Since the usages of the predicted novel sense `cell_3` are all referring to a constituent unit of a living organism, we may want to define this sense as, e.g., "A unit of a living organism". Then the submission for this target word will look like: 

| **Target word** | **sense_id** | **Definition**                  |
|-------------|----------|-----------------------------|
| _cell_        | `cell_3`   | "A unit of a living organism" |

The performance of the system is evaluated with BLEU and BertScore, comparing the generated definitions to the gold ones from manually annotated data.
The generated definitions are mapped to the gold ones by maximizing their BertScore similarity.

See [our evaluation scripts](https://github.com/ltgoslo/axolotl24_shared_task/tree/main/code/evaluation) for more details.

