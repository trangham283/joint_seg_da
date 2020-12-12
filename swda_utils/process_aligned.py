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

from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="/s0/ttmt001")

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

def postprocess_wordlevel(data_dir, split):
    filename = os.path.join(data_dir, split + "_aligned.tsv")
    df = pd.read_csv(filename, sep="\t")
    list_row = []
    da_lengths = []
    # order by dialog
    for filenum, df_file in df.groupby('filenum'):
        df_sorted_sents = df_file.sort_values('turn_id', kind='mergesort')
        # setting sort=False preserves turn orders
        for sent_id,df_sorted in df_sorted_sents.groupby('sent_id', sort=False):
            ms_tokens = df_sorted.ms_token.tolist()
            da_tokens = df_sorted.da_token.tolist()
            da_len = len(da_tokens)
            da_lengths.append(da_len)
            list_row.append({
                'filenum': filenum,
                'da_speaker': df_sorted.da_speaker.values[0],
                'true_speaker': df_sorted.true_speaker.values[0],
                'turn_id': df_sorted.turn_id.values[0],
                'sent_id': sent_id,
                'da_label': df_sorted.da_label.values[0],
                'da_sent': ' '.join(da_tokens),
                'ms_sent': ' '.join(ms_tokens),
                'start_times': df_sorted.start_time.tolist(),
                'end_times': df_sorted.end_time.tolist()
                })
    print("Max len in split: ", max(da_lengths))
    # train: 81; dev: 49; test: 60
    return pd.DataFrame(list_row), da_lengths

def get_bert_times(row):
    copy_num = len(row.bert_toks)
    start = [row.start_time]*copy_num
    end = [row.end_time]*copy_num
    return start, end

def get_bert_da_labels(row):
    copy_num = len(row.bert_toks)
    labels = [row.da_label]*copy_num
    sent_ids = [row.sent_id]*copy_num
    return labels, sent_ids

def postprocess_turnlevel(data_dir, split, data_origin="bert"):
    if data_origin == "bert":
        suffix = "_aligned.tsv"
    else:
        suffix = "_asr.tsv"
    filename = os.path.join(data_dir, split + suffix)
    df = pd.read_csv(filename, sep="\t")

    # TODO: create a column for grouping: filenum+asrhyp
    # just use 1-best hyp for now
    if data_origin == 'asr':
        df['asr_hyp'] = df.sent_id.apply(lambda x: int(x.split('-')[-1]))
        dfs = []
        for orig_id, df_sent in df.groupby('orig_id'):
            this_df = df_sent[df_sent.asr_hyp == df_sent.asr_hyp.min()]
            dfs.append(this_df)
        df = pd.concat(dfs).reset_index()

    da_lengths = []
    sessions = {}
    for filenum, df_file in df.groupby('filenum'):
        list_row = []
        df_sorted_sents = df_file.sort_values('turn_id', kind='mergesort')
        for turn_id, df_turn in df_sorted_sents.groupby('turn_id'):
            df_sorted = df_turn[df_turn.da_token != "<MISSED>"]
            if len(df_sorted) < 1:
                print("empty turn:", filenum, turn_id)
                continue
            df_sorted['bert_toks'] = df_sorted.da_token.apply(
                    bert_tokenizer.tokenize)
            df_sorted['bert_start'] = df_sorted.apply(lambda x: 
                    get_bert_times(x)[0], axis=1)
            df_sorted['bert_end'] = df_sorted.apply(lambda x: 
                    get_bert_times(x)[1], axis=1)
            bert_starts = df_sorted.bert_start.tolist()
            bert_starts = [item for x in bert_starts for item in x]
            bert_ends = df_sorted.bert_end.tolist()
            bert_ends = [item for x in bert_ends for item in x]
            joint_labels = []
            turn_tokens = []
            sent_ids = []
            for sent_id, sent_df in df_sorted.groupby('sent_id'):
                bert_tokens = sent_df.bert_toks.tolist()
                bert_tokens = [item for x in bert_tokens for item in x]
                turn_tokens += bert_tokens
                if data_origin == "asr":
                    dialog_act = "asr"
                else:
                    dialog_act = sent_df.da_label.values[0]
                joint_labels += ["I"]*(len(bert_tokens) - 1) + ["E_"+dialog_act]
                sent_ids += [sent_id]*len(bert_tokens)
            # NOTE: needed to convert to int() here bc of JSON
            if data_origin == "asr":
                speaker = df_sorted.speaker.values[0]
            else:
                speaker = df_sorted.true_speaker.values[0]
            list_row.append({
                'filenum': int(filenum),
                'speaker': speaker,
                'turn_id': int(df_sorted.turn_id.values[0]),
                'sent_ids': sent_ids,
                'joint_labels': joint_labels,
                'da_turn': turn_tokens,
                'start_times': bert_starts,
                'end_times': bert_ends
                })
            sessions[filenum] = list_row
            da_len = len(turn_tokens)
            da_lengths.append(da_len)
    print("Max turn len in split: ", max(da_lengths))
    # train: 81; dev: 49; test: 60
    return sessions

def main():
    """main function"""
    pa = argparse.ArgumentParser(description='combine tokens into turns')
    pa.add_argument('--data_dir', help="data path", default="../../data/joint")
    pa.add_argument('--split', default="dev", help="data split")
    pa.add_argument('--data_origin', default="bert", help="gold or asr")

    args = pa.parse_args()
    data_dir = args.data_dir
    split = args.split
    data_origin = args.data_origin

    #split_df, lengths = postprocess(data_dir, split)
    #outname = os.path.join(data_dir, split + '_aligned_dialogs.tsv')
    #lenlog = os.path.join(data_dir, split + '_lengths.json')
    #with open(lenlog, 'w') as fout:
    #    json.dump(lengths, fout, indent=2)

    #split_df, lengths = postprocess_wordlevel(data_dir, split)
    #outname = os.path.join(data_dir, split + '_aligned_dialogs_wordlevel.tsv')
    #split_df.to_csv(outname, sep="\t", index=False)
    
    outname = os.path.join(data_dir, split + '_' + data_origin + '_turns.json')
    sessions = postprocess_turnlevel(data_dir, split, data_origin)
    with open(outname, "w") as f:
        json.dump(sessions, f)
    
    exit(0)

if __name__ == '__main__':
    main()

