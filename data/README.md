# AXOLOTL data

The `russian`, `finnish` and `german` directories contain the data for the AXOLOTL shared task. 

This document is intended to give a brief description of the data format, along with some points worth taking notice of.

## Finnish

1. `axolotl.train.fi.tsv`: a training set of 45 897 old literary Finnish contexts of usages dated _before_ 1700, and 47 242 old literary Finnish contexts of usages dated _after_ 1700,
each mapped to a target headword, its sense ID, and a definition gloss in modern Finnish.
2. `axolotl.dev.fi.tsv`: a development set of 3 203 old literary Finnish contexts of usages dated _before_ 1700, and 3 351 old literary Finnish contexts of usages dated _after_ 1700,
each mapped to a target headword, its sense ID, and a definition gloss in modern Finnish.
3. `axolotl.test.fi.gold.tsv`: a test set of 3 461 old literary Finnish contexts of usages dated _before_ 1700, and 3 264 old literary Finnish contexts of usages dated _after_ 1700,
each mapped to a target headword, its sense ID, and a definition gloss in modern Finnish.

The data source is Vanhan kirjasuomen sanakirja [Dictionary of Old Literary Finnish]. 2023. Helsinki: Kotimaisten kielten keskuksen verkkojulkaisuja 38. Digital resource. URL: [https://kaino.kotus.fi/vks](https://kaino.kotus.fi/vks). XML version: [https://kaino.kotus.fi/lataa/vks.zip](https://kaino.kotus.fi/lataa/vks.zip). Last update 24.11.2023. Accessed 24.11.2023.

## Russian

1. `axolotl.train.ru.tsv`: a training set of 1 912 Russian usages dated XIX century and 4 581 Russian modern usages, each mapped to a target headword, its sense ID, and a definition gloss in modern Russian (unlike in the Finnish data, some usage examples for the XIX century are missing or noisy)
2. `axolotl.dev.ru.tsv`: a development set of 421 Russian usages dated XIX century and 1 605 Russian modern usages, each mapped to a target headword, its sense ID, and a definition gloss in modern Russian (unlike in the Finnish data, some usage examples for the XIX century are missing or noisy)
3. `axolotl.test.ru.gold.tsv`: a test set of 424 Russian usages dated XIX century and 1 702 Russian modern usages, each mapped to a target headword, its sense ID, and a definition gloss in modern Russian (unlike in the Finnish data, some usage examples for the XIX century are missing or noisy)

The data sources are Dahl V. (1909) Explanatory Dictionary of the Living Great Russian Language ed. by Boduen de Kurtene [Tolkovy slovar zhivogo velikorusskogo yazyka, pod red. I. A. Boduena de Kurtene] Saint Petersburg. for the old data and [CODWOE](https://aclanthology.org/2022.semeval-1.1/) (Mickus et al., SemEval 2022)  for the new data. We have used the [TEI-encoded version of the Dahl's Explantatory Dictionary](https://www.dialog-21.ru/media/4551/mikhaylovsaplusshershnevadm.pdf)(Mikhaylov and Shershneva, 2018) created by Liubov Polianskaia and Elena Shakurova.

## German

German was a surprise language introduced in the test phase only in order to evaluate the models' ability to handle an unseen language.

1. `axolotl.test.surprise.gold.tsv`: a test set of 584 old German usages and 568 new German usages.

The data source is [DWUG DE Sense](https://www.ims.uni-stuttgart.de/forschung/ressourcen/experiment-daten/dwug-de-sense/) (Dominik Schlechtweg. 2023. [Human and Computational Measurement of Lexical Semantic Change](http://dx.doi.org/10.18419/opus-12833). PhD thesis. University of Stuttgart).

## Data structure

Training and development sets are structured as tab-separated-values (TSV) files. 
Every row corresponds to one usage example (think sentences).

The files contain 9 named columns, as follows:

- `usage_id`: usage ID, unique across all AXOLOTL data, templated as `<dataset>_<language>_<row number>`, e.g. `dev_ru_0`
- `word`: target headword 
- `orth`: the target word in an old spelling (if applicable)
- `sense_id`: unique ID of the sense in which the target headword is used in the current example usage 
- `gloss`: definition of the sense
- `example`: usage example of the headword, usually a sentence, but can also be longer or shorter 
- `indices_target_token`: automatically produced character offsets for the headword in its usage example of usage, if applicable 
- `date` a coarse-grained date of attestation of the usage example (year, if applicable)
- `period`: indicator of the usage example belonging to the first ("old") or the second ("new") time period; 
thus, can take either the value of "**old**" or the value of "**new**". 

The test splits in the `test` folder have `sense_id` and `gloss` fields empty for the usages from the "new" time period.
The participants' task is to fill in the `sense_id` values in Subtask 1 and the definitions for the novel senses in Subtask 2. 

The Finnish datapoints are split depending on whether examples of usage were taken in the "old" time period (1543 to 1650) or the "new" time period (1700 to 1750). For Russian, "old" is approximately XIX century and "new" is modern language (approximately after 1950).

Note that headwords (target words) are split-specific, that is, a target word occurring in the training set, 
will never occur in the development and test sets, and vice versa.
