# AXOLOTL'24 - Test phase results

This directory contain the results for the test phase of the AXOLOTL'24 shared task

- `subtask1_leaderboard_ARI.tsv`: scores for Subtask 1 evaluated with ARI
- `subtask1_leaderboard_F1.tsv`: scores for Subtask 1 evaluated with F1
- `subtask2_leaderboard.tsv`: scores for Subtask 2 evaluated with BLEU and BERTScore.

The teams are ranked by the average of their scores across all three languages including the surprise language (Finnish, Russian and German).
It's the first `Fi-Ru-De` column.
For convenience, we also show the average scores without the surprise language in the `Fi-Ru` column.
The rest of the columns are results for specific languages.

## Metrics

For the Subtask 1, we keep separate leaderboards for ARI and F1, since these metrics focus on very different aspects of the task, and it does not make sense to average across them.

For the Subtask 2, we average across BLEU and BERTScore, since they aim at measuring the same aspects of the task.

## Codalab

You can also check the Codalab leaderboards which are now public:

- **[Codalab competition for Subtask 1](https://codalab.lisn.upsaclay.fr/competitions/18009)**
- **[Codalab competition for Subtask 2](https://codalab.lisn.upsaclay.fr/competitions/18008)**

