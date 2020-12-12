#!/usr/bin/env python3
"""
"""

import os
import sys, json
import argparse
import re
import pandas as pd
import glob
import numpy as np
import jiwer

from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="/s0/ttmt001")

def get_bert_times(row):
    copy_num = len(row.bert_toks)
    start = [row.start_time]*copy_num
    end = [row.end_time]*copy_num
    return start, end

def compute_wer(data_dir, split):
    overall_ref = ""
    overall_hyp = ""
    filename = os.path.join(data_dir, split + '_aligned.tsv')
    df_orig = pd.read_csv(filename, sep="\t")
    df_orig['orig_id'] = df_orig.apply(lambda row: 
            '{}_{}_{}'.format(row.filenum, row.true_speaker, 
                str(row.turn_id).zfill(4)), axis=1)
    filename = os.path.join(data_dir, split + '_asr.tsv')
    df_asr = pd.read_csv(filename, sep="\t")
    df_asr['asr_hyp'] = df_asr.sent_id.apply(lambda x: int(x.split('-')[-1]))
    dfs = []
    for orig_id, df_sent in df_asr.groupby('orig_id'):
        this_df = df_sent[df_sent.asr_hyp == df_sent.asr_hyp.min()]
        dfs.append(this_df)
    df_asr = pd.concat(dfs).reset_index()
    wer_dict = {}
    orig_ids = set(df_orig.orig_id)
    for orig_id in orig_ids:
        turn_orig = df_orig[df_orig.orig_id == orig_id]
        turn_orig = turn_orig[turn_orig.da_token != "<MISSED>"]
        turn_asr = df_asr[df_asr.orig_id == orig_id]
        aref = turn_orig.da_token.tolist()
        ref = ' '.join(aref).replace(" '", "'")
        ahyp = turn_asr.da_token.tolist()
        hyp = ' '.join(ahyp)
        overall_ref += ref + " "
        overall_hyp += hyp + " "
        this_wer = jiwer.wer(hyp, ref)
        wer_dict[orig_id] = this_wer
    overall_wer = jiwer.wer(overall_ref, overall_hyp)
    return wer_dict, overall_wer


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

    list_row = []
    for filenum, df_file in df.groupby('filenum'):
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
    sessions = pd.DataFrame(list_row)
    return sessions

def main():
    """main function"""
    pa = argparse.ArgumentParser(description='combine tokens into turns')
    pa.add_argument('--data_dir', help="data path", default="../../data/joint")
    pa.add_argument('--split', default="dev", help="data split")

    args = pa.parse_args()
    data_dir = args.data_dir
    split = args.split

    orig_df = postprocess_turnlevel(data_dir, split, 'bert')
    orig_df['main_id'] = orig_df.apply(lambda row: 
            '{}_{}_{}'.format(row.filenum, row.speaker, 
                str(row.turn_id).zfill(4)), axis=1)
    orig_df.drop(columns=['filenum', 'speaker', 'turn_id', 'sent_ids'], 
            inplace=True)
    orig_df.rename(
            columns={'da_turn': 'da_turn_orig', 
                'start_times': 'start_times_orig', 
                'end_times': 'end_times_orig'}, inplace=True)
    asr_df = postprocess_turnlevel(data_dir, split, 'asr')
    asr_df['main_id'] = asr_df.apply(lambda row: 
            '{}_{}_{}'.format(row.filenum, row.speaker, 
                str(row.turn_id).zfill(4)), axis=1)
    asr_df.drop(columns=['filenum', 'speaker', 'turn_id', 
        'sent_ids', 'joint_labels'], inplace=True)
    asr_df.rename(
            columns={'da_turn': 'da_turn_asr', 
                'start_times': 'start_times_asr', 
                'end_times': 'end_times_asr'}, inplace=True)
    merged_df = pd.merge(orig_df, asr_df, on='main_id')
    wer_dict, overall_wer = compute_wer(data_dir, split)
    print("Overall WER: ", overall_wer)
    merged_df['wer'] = merged_df.main_id.apply(lambda x: wer_dict[x])
    outname = split + "_merged.tsv"
    merged_df.to_csv(outname, sep="\t", index=False)
    
    exit(0)

if __name__ == '__main__':
    main()

