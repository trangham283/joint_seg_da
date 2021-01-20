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
    pa.add_argument('--sp_model', default="sp10004", help="speech model")
    pa.add_argument('--tt_model', default="tt1000", help="text model")

    args = pa.parse_args()
    split = args.split
    sp_model = args.sp_model
    tt_model = args.tt_model

    filename = split + "_merged.tsv"
    merged_df = pd.read_csv(filename, sep="\t")
    for column in ['joint_labels', 'da_turn_orig', 'da_turn_asr']:
        merged_df[column] = merged_df[column].apply(convert_to_list)

    for column in ['start_times_orig', 'end_times_orig', 'start_times_asr', 'end_times_asr']:
        merged_df[column] = merged_df[column].apply(convert_to_list, turn_float=True)
    sp10004_df = get_results_df(sp_model, split, merged_df)
    print(sp10004_df.head(3))
    tt1000_df = get_results_df(tt_model, split, merged_df)
    print(tt1000_df.head(3))

    m1sp = batch_metrics(sp10004_df.labels.tolist(), sp10004_df.hyps_trans.tolist())
    m1tt = batch_metrics(tt1000_df.labels.tolist(), tt1000_df.hyps_trans.tolist())
    m2sp = batch_metrics_asr(sp10004_df.labels.tolist(), sp10004_df.hyps_trans.tolist())
    m2tt =  batch_metrics_asr(tt1000_df.labels.tolist(), tt1000_df.hyps_trans.tolist())
    m3sp = batch_metrics_asr(sp10004_df.labels.tolist(), sp10004_df.hyps_asr.tolist())
    m3tt = batch_metrics_asr(tt1000_df.labels.tolist(), tt1000_df.hyps_asr.tolist())
    columns1 = ["DSER", "DER", "Macro LWER", "Micro LWER"]
    columns2 = ["Macro SER", "Micro SER", "Macro NSER", "Micro NSER"]
    columns3 = ["Macro LER", "Micro LER", "Macro LWER", "Micro LWER"]
    columns4 = ["Macro DAER", "Micro DAER"]
    columns5 = ["SegWER", "JointWER", "Macro F1", "Micro F1"]

    print("Models:", tt_model, sp_model)
    print("\t".join(columns1))
    for result in [m1tt, m1sp]:
        print(f"{result['DSER']}\t{result['DER']}\t{result['Macro LWER']}\t{result['Micro LWER']}")
    print()            
    print("\t".join(columns5))
    for result in [m1tt, m1sp]:
        print(f"{result['strict segmentation error']}\t{result['strict joint error']}\t{result['Macro F1']}\t{result['Micro F1']}\t")
    print()            
    print("\t".join(columns4+columns3))
    for result in [m2tt, m2sp, m3tt, m3sp]:
        print(f"{result['Macro DAER']}\t{result['Micro DAER']}\t{result['Macro LER']}\t{result['Micro LER']}\t{result['Macro LWER']}\t{result['Micro LWER']}")
    print()
    print("\t".join(columns2))
    for result in [m2tt, m2sp, m3tt, m3sp]:
        print(f"{result['Macro SER']}\t{result['Micro SER']}\t{result['Macro NSER']}\t{result['Micro NSER']}")
    exit(0)

if __name__ == '__main__':
    main()

