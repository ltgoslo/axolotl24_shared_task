# Evaluation scripts

## Track 1

```commandline
python3 scorer_track1.py --gold GOLD_DATA.tsv --pred PREDICTIONS.tsv
```

### Baseline results
[baseline code](https://github.com/ltgoslo/axolotl24_shared_task/tree/main/code/baselines)

| Language     | ARI   | Macro F1 |
|--------------|-------|----------|
| Finnish, dev | 0.022 | 0.222    |
| Russian, dev | 0.073 | 0.280    |

## Track 2

```commandline
python3 scorer_track2.py PREDICTIONS.tsv GOLD_DATA.tsv OUTPUT_FILE.txt
```
