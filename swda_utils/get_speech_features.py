#!/user/bin/env python3

import os, sys, argparse
import glob, re
import json
import pandas as pd
import numpy as np
from numpy.polynomial import legendre
from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir="/s0/ttmt001")

fbank_dir = "/s0/ttmt001/acoustic_features_json/fbank_json"
fbank3_dir = "/s0/ttmt001/acoustic_features_json/fb3_json"
out_dir = "/s0/ttmt001/swda"
dur_file = "/homes/ttmt001/transitory/dialog-act-prediction/data/avg_word_stats.json"

with open(dur_file, 'r') as f:
    dur_stats = json.load(f)

def pause2cat(p):
    r = min(1, max(0, np.log(1 + p)))
    if np.isnan(p):
        cat = 2
    elif p == 0.0:
        cat = 1
    elif p <= 0.05:
        cat = 3
    elif p <= 0.2:
        cat = 4
    elif p <= 1.0:
        cat = 5
    else:
        cat = 6
    return cat, r

def get_fbank_stats(fbank, partition, sf_name):
    lo_A = np.empty((3,0))
    hi_A = np.empty((3,0))
    for sidx, eidx in partition:
        if sidx < 0: sidx = 0
        this_frames = fbank[:, sidx:eidx]
        if this_frames.size == 0:
            print("No frame in ", sf_name, sidx, eidx)
            continue
        e0 = [this_frames[0,:].min(), this_frames[0,:].max()]
        lo = np.sum(this_frames[1:21,:], axis=0)
        hi = np.sum(this_frames[21:,:], axis=0)
        elo = [min(lo), max(lo)]
        ehi = [min(hi), max(hi)]
        min_vec = np.array([e0[0], elo[0], ehi[0]]).reshape(3,1)
        max_vec = np.array([e0[1], elo[1], ehi[1]]).reshape(3,1)
        lo_A = np.hstack([lo_A, min_vec])
        hi_A = np.hstack([hi_A, max_vec])
    lo_A = np.min(lo_A, 1)
    hi_A = np.max(hi_A, 1)
    return lo_A, hi_A

def get_time_features(start_times, end_times, sent_toks):
    sframes = [int(np.floor(x*100)) for x in start_times]
    eframes = [int(np.ceil(x*100)) for x in end_times]
    partition = zip(sframes, eframes)
    word_dur = [y-x for x,y in zip(start_times, end_times)]
    sent_max = max(0.001, max(word_dur)) # prevent division by 0
    dur_max = [x/sent_max for x in word_dur]
    dur_mean = []
    sent_mean = max(0.001, np.mean(word_dur))  # prevent division by 0
    for i, x in enumerate(sent_toks):
        mean_wd = dur_stats[x]['mean'] if (x in dur_stats and dur_stats[x]['count'] > 5) else sent_mean
        dur_mean.append(word_dur[i]/mean_wd)
    pause_before = [6] # sentence boundary cat
    pause_after = []
    rp_before = [1.0] # raw pause
    rp_after = []
    for i in range(1, len(start_times)):
        p, rp = pause2cat(start_times[i] - end_times[i-1])
        pause_before.append(p)
        rp_before.append(rp)
    for i in range(len(start_times)-1):
        p, rp = pause2cat(start_times[i+1] - end_times[i])
        pause_after.append(p)
        rp_after.append(rp)
    pause_after.append(6)
    rp_after.append(1.0)
    return partition, pause_before, pause_after, rp_before, rp_after, dur_mean, dur_max 

def make_feats(args):
    data_dir = args.data_dir
    split = args.split
    suffix = args.suffix
    info_file = os.path.join(data_dir, split + suffix)
    with open(info_file, 'r') as f:
        sessions = json.load(f)

    out_sess = {}
    for filenum in sessions.keys():
        print(filenum)
        sess = sessions[filenum]
        out_sess[filenum] = []
        partitionsA = []
        partitionsB = []
        for turn_dict in sess:
            feats = get_time_features(turn_dict['start_times'], 
                    turn_dict['end_times'], turn_dict['da_turn'])
            speaker = turn_dict['speaker']
            partition = list(feats[0])
            turn_dict.update({
                'partition': partition,
                'pause_before': feats[1], 'pause_after': feats[2],
                'rp_before': feats[3], 'rp_after': feats[4],
                'dur_mean': feats[5], 'dur_max': feats[6],
                'sent_ids': sorted(set(turn_dict['sent_ids']))
                })
            # might as well make the summarized version of fbank feats
            if speaker == 'A':
                partitionsA += partition
            else:
                partitionsB += partition
            out_sess[filenum].append(turn_dict)

        # This only needs to be done once:
        for speaker in ['A', 'B']:
            if speaker == 'A':
                fbank_file = os.path.join(fbank_dir, 'sw' + filenum + '-A.json')
                out_file = os.path.join(fbank3_dir, 'sw' + filenum + '-A.json')
                partition = partitionsA
            else:
                fbank_file = os.path.join(fbank_dir, 'sw' + filenum + '-B.json')
                out_file = os.path.join(fbank3_dir, 'sw' + filenum + '-B.json')
                partition = partitionsB
            with open(fbank_file, 'r') as f:
                fbank = json.load(f)
            fbank = np.array(fbank).T
            minE, maxE = get_fbank_stats(fbank, partition, filenum+"+"+speaker)
            e0 = fbank[0,:]
            elo = np.sum(fbank[1:21,:], axis=0)
            ehi = np.sum(fbank[21:,:], axis=0)
            efeats = np.vstack([e0, elo, ehi])
            hilo = maxE.reshape(3,1) - minE.reshape(3,1)
            efeats = (efeats - minE.reshape(3,1))/hilo

            with open(out_file, 'w') as f:
                outfeats = efeats.T
                outfeats = [list(x) for x in outfeats]
                json.dump(outfeats, f, indent=2)

    sessname = os.path.join(out_dir, split + "_bert_time_data.json")

    with open(sessname, 'w') as f:
        json.dump(out_sess, f)
    return

# TODO/FIXME: split 10 hypotheses into different keys 
def make_asr_feats(args):
    data_dir = args.data_dir
    split = args.split
    suffix = args.suffix
    info_file = os.path.join(data_dir, split + suffix)
    with open(info_file, 'r') as f:
        sessions = json.load(f)

    out_sess = {}
    for filenum in sessions.keys():
        print(filenum)
        sess = sessions[filenum]
        out_sess[filenum] = []
        partitionsA = []
        partitionsB = []
        for turn_dict in sess:
            feats = get_time_features(turn_dict['start_times'], 
                    turn_dict['end_times'], turn_dict['da_turn'])
            speaker = turn_dict['speaker']
            partition = list(feats[0])
            turn_dict.update({
                'partition': partition,
                'pause_before': feats[1], 'pause_after': feats[2],
                'rp_before': feats[3], 'rp_after': feats[4],
                'dur_mean': feats[5], 'dur_max': feats[6],
                'sent_ids': sorted(set(turn_dict['sent_ids']))
                })
            # might as well make the summarized version of fbank feats
            if speaker == 'A':
                partitionsA += partition
            else:
                partitionsB += partition
            out_sess[filenum].append(turn_dict)
    sessname = os.path.join(out_dir, split + "_asr_time_data.json")
    with open(sessname, 'w') as f:
        json.dump(out_sess, f)

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description = \
        "Preprocess segment files to get features")
    pa.add_argument('--split', type=str, \
        default="dev", help="split")
    pa.add_argument('--data_dir', 
        default="/homes/ttmt001/transitory/dialog-act-prediction/data/joint")

    args = pa.parse_args()
    #make_feats(args) 
    make_asr_feats(args) 
    exit(0)
 
