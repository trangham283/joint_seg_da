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
from difflib import SequenceMatcher
from metric_helpers import *


ref_dir = "/homes/ttmt001/transitory/dialog-act-prediction/data/joint/ref_out"
asr_dir = "/homes/ttmt001/transitory/dialog-act-prediction/data/joint/asr_out"



def convert_to_list(this_str, turn_float=False):
    this_str = this_str.replace('[', '').replace(']','')
    this_str = this_str.replace("'", "").replace(",","").split()
    if turn_float:
        this_str = [float(x) for x in this_str]
    return this_str

def get_results_df(model_name, split_name, merged_df):
    suffix = split_name.upper() + '_' +  model_name + '.res'

    trans_file = os.path.join(ref_dir, suffix)
    asr_file = os.path.join(asr_dir, suffix)

    trans_df = pd.read_csv(trans_file, sep="\t")
    asr_df = pd.read_csv(asr_file, sep="\t")
    asr_df.rename(columns={'PREDS': 'PREDS_ASR'}, inplace=True)
    asr_df['PREDS_ASR'] = asr_df.PREDS_ASR.apply(lambda x: x.replace(" </t>", ""))
    preds_df = trans_df.join(asr_df)
    preds_df['labels'] = preds_df.LABELS.apply(lambda x: x.split())
    preds_df['hyps_trans'] = preds_df.PREDS.apply(lambda x: x.split())
    preds_df['hyps_asr'] = preds_df.PREDS_ASR.apply(lambda x: x.split())
    preds_df.rename(columns={'TURN_ID': 'main_id'}, inplace=True)
    preds_df.drop(columns=['LABELS', 'PREDS', 'PREDS_ASR'], inplace=True)
    res_df = pd.merge(preds_df, merged_df, on='main_id')

    results = res_df.apply(lambda row: instance_metrics(row.labels, row.hyps_trans), axis=1)
    results_asr = res_df.apply(lambda row: instance_metrics_asr(row.labels, row.hyps_asr), axis=1)
    results2 = res_df.apply(lambda row: instance_metrics_asr(row.labels, row.hyps_trans), axis=1)

    res_df['DSER'] = [x['DSER'] for x in results.tolist()]
    res_df['DER'] = [x['DER'] for x in results.tolist()]
    res_df['LWER_trans'] = [x['LWER'] for x in results.tolist()]
    res_df['LER_trans'] = [x['LER'] for x in results2.tolist()]
    res_df['SER_trans'] = [x['SER'] for x in results2.tolist()]
    res_df['NSER_trans'] = [x['NSER'] for x in results2.tolist()]
    res_df['DAER_trans'] = [x['DAER'] for x in results2.tolist()]

    res_df['LWER_asr'] = [x['LWER'] for x in results_asr.tolist()]
    res_df['LER_asr'] = [x['LER'] for x in results_asr.tolist()]
    res_df['SER_asr'] = [x['SER'] for x in results_asr.tolist()]
    res_df['NSER_asr'] = [x['NSER'] for x in results_asr.tolist()]
    res_df['DAER_asr'] = [x['DAER'] for x in results_asr.tolist()]

    return res_df

def main():
    """main function"""
    pa = argparse.ArgumentParser(description='combine tokens into turns')
    pa.add_argument('--split', default="dev", help="data split")
    pa.add_argument('--model_name', default="sp10004", help="data split")

    args = pa.parse_args()
    data_dir = args.data_dir
    split = args.split
    model_name = args.model_name

    filename = split + "_merged.tsv"
    merged_df = pd.read_csv(filename, sep="\t")
    for column in ['joint_labels', 'da_turn_orig', 'da_turn_asr']:
        merged_df[column] = merged_df[column].apply(convert_to_list)

    for column in ['start_times_orig', 'end_times_orig', 'start_times_asr', 'end_times_asr']:
        merged_df[column] = merged_df[column].apply(convert_to_list, turn_float=True)
    sp10004_df = get_results_df("sp10004", split, merged_df)
    tt1000_df = get_results_df("tt1000", split, merged_df)


    batch_metrics(sp10004_df.labels.tolist(), sp10004_df.hyps_trans.tolist())
    batch_metrics(tt1000_df.labels.tolist(), tt1000_df.hyps_trans.tolist())
    batch_metrics_asr(sp10004_df.labels.tolist(), sp10004_df.hyps_asr.tolist())
    batch_metrics_asr(tt1000_df.labels.tolist(), tt1000_df.hyps_asr.tolist())

    
    exit(0)

if __name__ == '__main__':
    main()

