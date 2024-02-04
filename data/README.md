# AXOLOTL - Finnish and Russian training and development data

The `russian` and `finnish` directories contain the training and development data for the AXOLOTL shared task. 
**The test sets will be published on March 25, 2024**.

This document is intended to give a brief description of the data format, along with some points worth taking notice of.

## Finnish

1. `axolotl.train.fi.tsv`: a training set of 45 897 old literary Finnish contexts of usages dated _before_ 1700, and 47 242 old literary Finnish contexts of usages dated _after_ 1700,
each mapped to a target headword, its sense ID, and a definition gloss in modern Finnish.
2. `axolotl.dev.fi.tsv`: a development set of 3 203 old literary Finnish contexts of usages dated _before_ 1700, and 3 351 old literary Finnish contexts of usages dated _after_ 1700,
each mapped to a target headword, its sense ID, and a definition gloss in modern Finnish.

## Russian

1. `axolotl.train.ru.tsv`: a training set of 2 514 Russian usages dated XIX. century and 5 478 Russian modern usages,
each mapped to a target headword, its sense ID, and a definition gloss in XIX. century (if many modern senses were mapped to the same old one) or modern Russian (if exactly one sense was mapped to an old one)
(unlike in the Finnish data, some usage examples for the XIX. century are missing or noisy)
2. `axolotl.dev.ru.tsv`: a development set of 530 Russian usages dated XIX. century and 1 939 Russian modern usages,
each mapped to a target headword, its sense ID, and a definition gloss in XIX. century (if many modern senses were mapped to the same old one) or modern Russian (if exactly one sense was mapped to an old one)
(unlike in the Finnish data, some usage examples for the XIX. century are missing or noisy)


## Training and development sets structure

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
thus, can take either the value of "old" or the value of "new". 

When the test splits will be published, their `sense_id` and `gloss` fields will be empty for the usages from the "new" time period.
The participants will have to fill in the `sense_id` values in Subtask 1 and the definitions for the novel senses in Subtask 2. 

The Finnish datapoints are split depending on whether examples of usage were taken in the "old" time period (1543 to 1650) or the "new" time period (1700 to 1750). For Russian, "old" is approximately XIX century and "new" is modern language (approximately after 1950).

Note that headwords (target words) are split-specific, that is, a target word occurring in the training set, 
will never occur in the development and test sets, and vice versa.

## Expected future updates

Future updates of the provided Finnish dataset will include manually verified character offsets for the headword in its example of usage in the development sets (both period). 
The corresponding version update is expected to be released on February 9th, 2024.

Russian datasets will be further cleaned and filtered throughout February. The data format will remain the same.
