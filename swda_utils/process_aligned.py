#!/usr/bin/env python3
"""
Combines aligned time data into format readable by da_main_*.py and
correponding dataset readers
"""

import os
import sys, json
import argparse
import re
import pandas as pd
import glob
import numpy as np


def postprocess(data_dir, split):
    filename = os.path.join(data_dir, split + "_aligned.tsv")
    df = pd.read_csv(filename, sep="\t")
    list_row = []
    da_lengths = []
    # order by dialog
    for filenum, df_file in df.groupby('filenum'):
        df_sorted_turns = df_file.sort_values('turn_id', kind='mergesort')
        for turn_id, df_turn in df_sorted_turns.groupby('turn_id'):
            df_sorted = df_turn.sort_values('start_time', kind='mergesort')
            ms_tokens = df_sorted.ms_token.tolist()
            ms_tokens = [x for x in ms_tokens if isinstance(x, str)]
            da_tokens = df_sorted.da_token.tolist()
            da_tokens = [x for x in da_tokens if x != "<MISSED>"]
            da_len = len(da_tokens)
            da_lengths.append(da_len)
            list_row.append({
                'filenum': filenum,
                'da_speaker': df_sorted.da_speaker.values[0],
                'true_speaker': df_sorted.true_speaker.values[0],
                'turn_id': turn_id,
                'da_label': df_sorted.da_label.values[0],
                'da_sent': ' '.join(da_tokens),
                'ms_sent': ' '.join(ms_tokens),
                'start_time': df_sorted.start_time.values[0],
                'end_time': df_sorted.end_time.values[-1]
                })
    print("Max len in split: ", max(da_lengths))
    # train: 81; dev: 49; test: 60
    return pd.DataFrame(list_row), da_lengths

def main():
    """main function"""
    pa = argparse.ArgumentParser(description='combine tokens into turns')
    pa.add_argument('--data_dir', help="data path", default="../../data/joint")
    pa.add_argument('--split', default="dev", help="data split")

    args = pa.parse_args()
    data_dir = args.data_dir
    split = args.split

    split_df, lengths = postprocess(data_dir, split)
    outname = os.path.join(data_dir, split + '_aligned_dialogs.tsv')
    split_df.to_csv(outname, sep="\t", index=False)
    lenlog = os.path.join(data_dir, split + '_lengths.json')
    with open(lenlog, 'w') as fout:
        json.dump(lengths, fout, indent=2)

    exit(0)

if __name__ == '__main__':
    main()

