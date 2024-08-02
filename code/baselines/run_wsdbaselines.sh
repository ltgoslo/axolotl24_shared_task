#!/bin/bash

if ! [ -f ../../data/german/axolotl.test.ge.gold.tsv  ]; then
  echo "Downloading german test set..."
  cd ../../data/german
  curl -o dwug_de_sense.zip https://zenodo.org/records/8197553/files/dwug_de_sense.zip?download=1
  unzip dwug_de_sense.zip
  python surprise.py --dwug_path dwug_de_sense/
  mv axolotl.test.surprise.gold.tsv axolotl.test.ge.gold.tsv
  cd -
fi

for lang in finnish russian german; do
for baseline in random_old_sense mfs_old_sense; do

seq 100 | parallel -j 100 \
 "python3 wsdbaselines_track1.py ../../data/${lang}/axolotl.test.${lang:0:2}.gold.tsv pred_{}.tsv ${baseline} --seed={} &&\
  python3 ../evaluation/scorer_track1.py --gold ../../data/${lang}/axolotl.test.${lang:0:2}.gold.tsv --pred pred_{}.tsv --output scores_{}.tsv"

cat scores_*.tsv >${baseline}.${lang}.scores.tsv
rm pred*.tsv scores*.tsv

done
done

python wsdbaselines_track1_table.py