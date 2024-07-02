## German

German was a surprise language introduced in the test phase only in order to evaluate the models' ability to handle an unseen language.

The data source is [DWUG DE Sense](https://www.ims.uni-stuttgart.de/forschung/ressourcen/experiment-daten/dwug-de-sense/) . Please cite them as "Dominik Schlechtweg. 2023. [Human and Computational Measurement of Lexical Semantic Change](http://dx.doi.org/10.18419/opus-12833). PhD thesis. University of Stuttgart".

In order to convert it to the axolotl format:

1. [Download](https://zenodo.org/records/8197553) and unpack `dwug_de_sense.zip` .

2. 
```commandline
python surprise.py <path to dwug_de_sense/>
```

3. This will create `axolotl.test.surprise.gold.tsv`: a test set of 584 old German usages and 568 new German usages.