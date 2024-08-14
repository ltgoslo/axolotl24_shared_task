from pathlib import Path

import numpy as np
import pandas as pd


def main():
    dfs = []
    for p in Path('./baseline_scores').glob('*.scores.tsv'):
        mdf = pd.read_csv(p, sep=': +', header=None, engine='python')
        mdf.rename(columns={0: 'metric', 1: 'value'}, inplace=True)
        ff = p.name.split('.')
        mdf['method'] = ff[0]
        mdf['lang'] = ff[1]
        mdf['part'] = ff[2]
        dfs.append(mdf)

    mdf = pd.concat(dfs, ignore_index=True)
    results = mdf.groupby(['method', 'lang', 'part', 'metric']).value.agg(
        lambda r: f'{np.mean(r):.3f}+-{np.std(r):.3f} ({len(r)})'
    ).reset_index()

    results = results.pivot_table(index=['method', 'part'], columns=['metric', 'lang'], values='value',
                                  aggfunc=lambda r: ' '.join(r))
    results.columns = [f'{c2}-{c1}' for c1, c2 in results.columns]
    results.to_csv('wsdbaselines.scores.tsv', sep='\t', index=True)

if __name__ == "__main__":
    main()
