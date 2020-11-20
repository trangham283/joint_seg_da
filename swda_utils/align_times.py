#!/usr/bin/env python3
"""
Does time alignments for utterances in SWBD and MS-State datasets
"""
import os
import sys
import argparse
import re
import string
import pandas as pd
import glob
import json

from difflib import SequenceMatcher
from collections import Counter

# nonverbal_reg_exp = r"\<+[^\<\>]+\>+|[a-zA-Z]+_[a-zA-Z]+"
nonverbal_reg_exp_list = [re.compile(r"\<+[^\<\>]+\>+"), re.compile(r"\w+_\w+")]
nonverbal_token = "NONVERBAL"
skip_punt_reg_exp = re.compile(r"[#\-\(\)]+")

# MS transcript column names
ms_cols = ['utt_id', 'start_time', 'end_time', 'word']

SP2_FILES = ['3138', '3399', '3850']
SP6_FILES = ['2423', '2528', '2568', '2788', '2594', '3170', '3259', '3469', 
        '3275', '3281', '3320', '3330', '3419', '3426', '3521', '3586', '3747',
        '4327', '4342', '4347', '4483', '4603', '4608', '4611', '4618', '4643', 
        '4644', '4659', '4721', '4726', '4880']
SP6_NOTB = ['2568', '3281', '3330', '3419', '3586', '3747', '4880']
SP6_NOTA = ['2423', '2594', '2788']
TAB_FILES = ['2579', '2776', '2927', '3187', '3202', '3226', '3266', '3362', 
        '3646', '3751', '3936', '4082', '4133', '4628']

# Switched sides
SWITCHED = set(['3129', '2006', '2010', '2027', '2064', '2072', '2073', '2110', 
        '2130', '2171', '2177', '2235', '2247', '2262', '2279', '2290', '2292',
        '2303', '2305', '2339', '2366', '2372', '2405', '2476', '2485', '2501',
        '2514', '2521', '2527', '2533', '2539', '2543', '2566', '2576', '2593',
        '2616', '2617', '2627', '2631', '2658', '2684', '2707', '2789', '2792',
        '2794', '2844', '2854', '2858', '2913', '2930', '2932', '2954', '2955', 
        '2960', '2963', '2968', '2970', '2981', '2983', '2994', '2999', '3000',
        '3012', '3013', '3018', '3039', '3040', '3050', '3061', '3077', '3088', 
        '3096', '3130', '3131', '3136', '3138', '3140', '3142', '3143', '3144',
        '3146', '3148', '3154', '3405'])

MAP_MS = {'[noise]': nonverbal_token, 
        '[laughter]': nonverbal_token, 
        '[vocalized-noise]': nonverbal_token}

def unroll_toks(df_in):
    list_row = []
    for i, row in df_in.iterrows():
        tokens = row.normed_text.split()
        for t in tokens:
            list_row.append({'turn_id': row.turn_id, \
                    'speaker': row.speaker, \
                    'da_token': t, \
                    'sent_id': row.sentence_id, \
                    'da_label': row.da_label})
    df = pd.DataFrame(list_row)
    return df

def norm_ms(tok):
    orig = tok[:]
    if tok in MAP_MS: 
        tok = MAP_MS[tok]
    if "/" in tok:
        idx = tok.index("/")
        tok = tok[1:idx]
    if '[laughter-' in tok:
        tok = tok.replace('[laughter-', '').replace(']', '')
    if tok.endswith('_1'):
        tok = tok.replace('_1', '')
    if tok.count('[') > 1:
        tok = tok[tok.index(']')+1:]
    if tok.endswith(']-'):
        idx = tok.index('[')
        tok = tok[:idx]
    if tok.startswith('-['):
        idx = tok.index(']')
        tok = tok[idx+1:]
    if tok.endswith('-'):
        tok = tok[:-1]
    if "{" in tok:
        tok = tok.replace("{", "").replace("}", "")
    if "[" in tok and "]" in tok:
        idx1 = tok.index('[')
        idx2 = tok.index(']')
        tok = tok[:idx1]+tok[idx2+1:]
    if not tok: 
        print("Original", orig)
    return tok

# modify ms_toks_df to split by "-" and "'" but copy start and end times
# e.g. uh-huh --> uh huh
#      gonna/wanna --> going to / want to 
#      can't --> ca n't; cannot --> can not
#      it's --> it 's
# UPDATE: also remove <b_aside> and <e_aside> markers
SPLIT3 = {"cannot": ["can", "not"],
        "gonna": ["going", "to"],
        "wanna": ["want", "to"]
        }
def split_ms_toks(ms_toks_df):
    list_row = []
    for i, row in ms_toks_df.iterrows():
        utt_id = row.utt_id
        start_time = row.start_time
        end_time = row.end_time
        word = row.word
        word_norm = row.word_norm
        if word_norm in ['<b_aside>', '<e_aside>']:
            continue
        if word_norm in SPLIT3 or ("'" in word_norm and word_norm not in ["o'clock", "n't"] and word_norm[0] != "'"):
            if "n't" in word_norm:
                idx = word_norm.index("n't")
                tok1 = word_norm[:idx]
                tok2 = word_norm[idx:]
            elif "'" in word_norm:
                tok1, tok2 = word_norm.split("'")
                tok2 = "'"+tok2
            elif "-" in word_norm:
                tok1, tok2 = word_norm.split("-")
            else: # SPLIT3
                tok1, tok2 = SPLIT3[word_norm]
            list_row.append({'utt_id': utt_id,
                'start_time': start_time,
                'end_time': end_time,
                'word': word,
                'word_norm': tok1
                })
            list_row.append({'utt_id': utt_id,
                'start_time': start_time,
                'end_time': end_time,
                'word': word,
                'word_norm': tok2
                })
        elif "-" in word_norm:
            toks = word_norm.split("-")
            for tok in toks:
                list_row.append({'utt_id': utt_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'word': word,
                    'word_norm': tok
                    })
        else:
            list_row.append({'utt_id': utt_id,
                'start_time': start_time,
                'end_time': end_time,
                'word': word,
                'word_norm': word_norm
                })
    return pd.DataFrame(list_row)


# first pass check 
def check_dfs(da_unroll_df, ms_df, filenum, speaker, true_speaker, 
        skip_nonverbal=True):
    ms_toks_df = ms_df[ms_df['word'] != '[silence]']
    ms_toks_df.loc[:, 'word_norm'] = ms_toks_df.word.apply(norm_ms)
    if skip_nonverbal:
        ms_toks_df = ms_toks_df[ms_toks_df['word_norm'] != nonverbal_token]
        da_unroll_df = da_unroll_df[da_unroll_df['da_token'] != nonverbal_token]
        da_unroll_df = da_unroll_df[da_unroll_df['da_token'] != '...']
        da_unroll_df = da_unroll_df.reset_index()
    ms_toks_df = split_ms_toks(ms_toks_df)
    ms_toks_df = ms_toks_df.reset_index()
    ms_toks_df.loc[:, 'ms_tok_id'] = range(len(ms_toks_df))
    ms_toks_df.loc[:, 'su_id'] = ms_toks_df.utt_id.apply(lambda x: 
            int(x.split('-')[-1]))
    da_unroll_df.loc[:, 'da_tok_id'] = range(len(da_unroll_df))
    ms_toks_df = ms_toks_df.set_index('ms_tok_id')
    da_toks_df = da_unroll_df.set_index('da_tok_id')
    # .get_opcodes returns ops to turn a into b 
    ms_side = ms_toks_df.word_norm.tolist()
    ms_side = [x.lower() for x in ms_side if x != nonverbal_token]
    da_side = da_toks_df.da_token.tolist()
    sseq = SequenceMatcher(None, ms_side, da_side)
    for info in sseq.get_opcodes():
        tag, i1, i2, j1, j2 = info
        # checking for switched sides:
        # if tag != "equal" and (i2-i1>=10 or j2-j1>=10):
        if tag != "equal":
            #print(tag, i1, i2, j1, j2, len(ms_side), len(da_side))
            start_i = max(0, min(i1, len(ms_side) - 1))
            end_i = max(i1, min(i2 - 1, len(ms_side) - 1))
            start_j = max(0, min(j1, len(da_side) - 1))
            end_j = max(j1, min(j2 - 1, len(da_side) - 1))
            #print(start_i, end_i, start_j, end_j)
            ms_part = ms_toks_df.loc[start_i:end_i]
            da_part = da_toks_df.loc[start_j:end_j]
            prev_ms_turn = ms_toks_df.loc[max(i1-1,0)].su_id
            next_ms_turn = ms_toks_df.loc[min(i2, len(ms_side)-1)].su_id
            prev_da_turn = da_toks_df.loc[max(j1-1,0)].turn_id
            next_da_turn = da_toks_df.loc[min(j2, len(da_side)-1)].turn_id
            ms_turns = ms_part['su_id'].values
            da_turns = da_part['turn_id'].values
                
            start_time = ms_part.start_time.values[0]
            end_time = ms_part.end_time.values[-1]
            time_span = end_time - start_time
            #print("{}\t{}\t{}\t{}\t{}\t{}\t{:6.4f}\t{}\t{}\t{}\t{}".format(\
            #        filenum, speaker, true_speaker, tag, i2-i1, j2-j1, \
            #        time_span, len(ms_turns), len(da_turns), \
            #        ms_side[i1:i2], da_side[j1:j2]))

            # only look at problem cases
            if (len(ms_side[i1:i2]) > 1 and len(da_side[j1:j2]) > 1 and (j2-j1)!=(i2-i1)) and len(set(ms_turns)) != len(set(da_turns))  and len(set(ms_turns)) != 1 :
                if i2-i1 >= j2-j1: continue
                if len(set(da_turns)) == 1:
                    da_turn = da_turns[0]
                    ms_turn_counts = Counter(ms_turns)
                    ms_turn, turn_count = ms_turn_counts.most_common(1)[0]
                    if turn_count >= len(da_turns): continue
                print("{}\t{}\t{}\t{}\t{}\t{}\t{:6.4f}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(\
                            filenum, speaker, true_speaker, tag, i2-i1, j2-j1, \
                            time_span, prev_ms_turn, ms_turns, next_ms_turn, \
                            prev_da_turn, da_turns, next_da_turn, \
                            ms_side[i1:i2], da_side[j1:j2]))
            #    subseq = SequenceMatcher(None, ms_side[i1:i2], da_side[j1:j2])
            #    relign = subseq.get_opcodes()
            #    if len(relign) < 2: continue # means same alignment, not better
            #    for tag, i11, i22, j11, j22 in relign:
            #        if tag != 'equal':
            #            #print(i11, i22, j11, j22)
            #            print("\t", tag, ms_side[i1+i11:i1+i22], da_side[j1+j11:j1+j22])
            #            print("\t", ms_toks_df.loc[i1+i11:i1+i22].su_id.values, da_toks_df.loc[j1+j11:j1+j22].turn_id.values)
                
def get_align_checks(config, split):
    print("filenum\tspeaker\ttrue_speaker\toperation\tnum_ms\tnum_da\ttime_span\tprev_ms_turn\tms_su\tnext_ms_turn\tprev_da_turn\tda_su\tnext_da_turn\tms_side\tda_side")
    da_dir = os.path.join(config.task_data_dir, split)
    for filename in glob.glob(da_dir + '/*.tsv'):
        prefix = os.path.basename(filename)[2:4]
        filenum = os.path.basename(filename)[2:6]
        # debug done files
        # if int(filenum) < 4640: continue
        speaker = os.path.basename(filename)[7]
        true_speaker = speaker
        sub_ms_dir = os.path.join(config.ms_dir, prefix, filenum)
        if filenum in SWITCHED:
            if speaker == 'A':
                true_speaker = 'B'
            else:
                true_speaker = 'A'
        ms_file = os.path.join(sub_ms_dir,
                'sw' + filenum + true_speaker + '-ms98-a-word.text')
        if filenum in SP6_FILES:
            if (filenum in SP6_NOTA and speaker == 'A') or \
                (filenum in SP6_NOTB and speaker == 'B'):
                sep = " "
            else:
                sep = "     "
        elif filenum in SP2_FILES:
            sep = "  "
        elif filenum in TAB_FILES:
            sep = "\t"
        else:
            sep = " "
        ms_df = pd.read_csv(ms_file, sep=sep, names=ms_cols)
        da_df = pd.read_csv(filename, sep='\t')
        da_unroll_df = unroll_toks(da_df)
        #print(filenum, speaker)
        check_dfs(da_unroll_df, ms_df, filenum, speaker, true_speaker)


'''
Align/get time stamps for turns in DA set from MS set
let 
   prev_ms_turn = ms_side[i1-1].su_id
   next_ms_turn = ms_side[i2].su_id
   prev_da_side = da_side[j1-1].su_id
   next_da_side = da_side[j2].su_id
Cases to consider:
    1. Same tokens: transfer the start and end times of MS-tokens to DA-tokens
    2. Deletion: set da_token = "<MISSED>" and copy all other information
    3. Insertion: set ms_token = "<INSERTED>"
       for all tokens in da_side: 
       if prev_ms_turn==ms_side[i1].su_id and prev_da_side==da_side[j1].su_id:  
            start/end times = ms_side[i1-1]'s times
       else:
            start/end times = ms_side[i2]'s times
    4. len(ms_side[i1:i2]) == len(da_side[j2:j2]): transfer start and end time 
       for each token, assume they correspond one-to-one (this is a reasonable
       assumption based on anecdotal data)

    5. Cases where len(ms_side[i1:i2]) != len(da_side[j2:j2])
       if prev_ms_turn==ms_side[i1].su_id and prev_da_side==da_side[j1].su_id:
          transfer token times from the front
       else:
          transfer token times from the back
          anecdotally this works because difflib's align algo tends to be 
          front-greedy, so any remaining unaligned tokens seem to be 
          better aligned from the back
       a. len(ms_side[i1:i2]) > len(da_side[j2:j2]):
          works fine with the above
       b. len(ms_side[i1:i2]) < len(da_side[j2:j2]):
          copy start/end times of first (last) token if copied from back (front)

'''
def align_dfs(da_unroll_df, ms_df, filenum, speaker, true_speaker, 
        skip_nonverbal=True):
    list_row = []
    ms_toks_df = ms_df[ms_df['word'] != '[silence]']
    ms_toks_df.loc[:, 'word_norm'] = ms_toks_df.word.apply(norm_ms)
    if skip_nonverbal:
        ms_toks_df = ms_toks_df[ms_toks_df['word_norm'] != nonverbal_token]
        da_unroll_df = da_unroll_df[da_unroll_df['da_token'] != nonverbal_token]
        da_unroll_df = da_unroll_df[da_unroll_df['da_token'] != '...']
        da_unroll_df = da_unroll_df.reset_index()
    ms_toks_df = split_ms_toks(ms_toks_df)
    ms_toks_df = ms_toks_df.reset_index()
    ms_toks_df.loc[:, 'ms_tok_id'] = range(len(ms_toks_df))
    ms_toks_df.loc[:, 'su_id'] = ms_toks_df.utt_id.apply(lambda x: 
            int(x.split('-')[-1]))
    da_unroll_df.loc[:, 'da_tok_id'] = range(len(da_unroll_df))
    ms_toks_df = ms_toks_df.set_index('ms_tok_id')
    da_toks_df = da_unroll_df.set_index('da_tok_id')
    # .get_opcodes returns ops to turn a into b 
    ms_side = ms_toks_df.word_norm.tolist()
    ms_side = [x.lower() for x in ms_side]
    da_side = da_toks_df.da_token.tolist()
    #print(filenum, speaker, len(ms_side), len(da_side))
    sseq = SequenceMatcher(None, ms_side, da_side)
    for info in sseq.get_opcodes():
        tag, i1, i2, j1, j2 = info
        start_i = max(0, min(i1, len(ms_side) - 1))
        end_i = min(max(i1, i2 - 1), len(ms_side) - 1)
        start_j = max(0, min(j1, len(da_side) - 1))
        end_j = min(max(j1, j2 - 1), len(da_side) - 1)
        ms_part = ms_toks_df.loc[start_i:end_i]
        da_part = da_toks_df.loc[start_j:end_j]
        #if i2-i1 != j2-j1:
            #print(info, ms_side[i1:i2], da_side[j1:j2])
            #print(start_i, end_i, start_j, end_j)
        prev_ms_turn = ms_toks_df.loc[max(i1-1,0)].su_id
        next_ms_turn = ms_toks_df.loc[min(i2, len(ms_side)-1)].su_id
        prev_da_turn = da_toks_df.loc[max(j1-1,0)].turn_id
        next_da_turn = da_toks_df.loc[min(j2, len(da_side)-1)].turn_id
        if i2-i1 == j2-j1:
        # This is either equal or replace the same number of tokens
        # both cases transfer start/end times one by one
            for ms_idx, da_idx in zip(range(i1, i2), range(j1, j2)):
                da_token = da_part.loc[da_idx].da_token
                sent_id = da_part.loc[da_idx].sent_id
                turn_id = da_part.loc[da_idx].turn_id
                da_label = da_part.loc[da_idx].da_label
                start_time = ms_part.loc[ms_idx].start_time
                end_time = ms_part.loc[ms_idx].end_time
                ms_token = ms_part.loc[ms_idx].word_norm
                if not ms_token:
                    print(ms_part)
                list_row.append({
                    'filenum': filenum,
                    'da_speaker': speaker,
                    'true_speaker': true_speaker,
                    'ms_token': ms_token,
                    'da_token': da_token,
                    'sent_id': sent_id,
                    'turn_id': turn_id,
                    'da_label': da_label,
                    'start_time': start_time,
                    'end_time': end_time
                    })
        elif tag == 'delete':
            # j1 = j2; da_part should be empty but the way .loc works
            # has da_toks.loc[j1]
            da_token = "<MISSED>"
            da_idx = start_j
            sent_id = da_part.loc[da_idx].sent_id
            turn_id = da_part.loc[da_idx].turn_id
            da_label = da_part.loc[da_idx].da_label
            for ms_idx in range(i1, i2):
                start_time = ms_part.loc[ms_idx].start_time
                end_time = ms_part.loc[ms_idx].end_time
                ms_token = ms_part.loc[ms_idx].word_norm
                if not ms_token:
                    print(ms_part)
                list_row.append({
                    'filenum': filenum,
                    'da_speaker': speaker,
                    'true_speaker': true_speaker,
                    'ms_token': ms_token,
                    'da_token': da_token,
                    'sent_id': sent_id,
                    'turn_id': turn_id,
                    'da_label': da_label,
                    'start_time': start_time,
                    'end_time': end_time
                    })
        elif tag == "insert":
            # i1 = i2; ms_part should be empty but the way .loc works
            # has ms_toks.loc[i1]
            if next_ms_turn == ms_part.loc[end_i].su_id and \
                    next_da_turn == da_part.loc[end_j].turn_id:
                ms_idx = end_i 
            else:
                ms_idx = max(i1-1, 0) 
            ms_token = "<INSERTED>"
            start_time = ms_toks_df.loc[ms_idx].start_time
            end_time = ms_toks_df.loc[ms_idx].end_time
            for da_idx in range(j1, j2):
                da_token = da_part.loc[da_idx].da_token
                sent_id = da_part.loc[da_idx].sent_id
                turn_id = da_part.loc[da_idx].turn_id
                da_label = da_part.loc[da_idx].da_label
                if not ms_token:
                    print(ms_part)
                list_row.append({
                    'filenum': filenum,
                    'da_speaker': speaker,
                    'true_speaker': true_speaker,
                    'ms_token': ms_token,
                    'da_token': da_token,
                    'sent_id': sent_id,
                    'turn_id': turn_id,
                    'da_label': da_label,
                    'start_time': start_time,
                    'end_time': end_time
                    })
        else: #tag = "replace":
            len_toks = min(len(ms_part), len(da_part))
            range_ms = range(i1, i2)
            range_da = range(j1, j2)
            if prev_ms_turn == ms_part.loc[start_i].su_id \
                    and prev_da_turn == da_part.loc[start_j].turn_id:
                #print("front:", info, ms_side[i1:i2], da_side[j1:j2])
                for idx in range(len_toks):
                    ms_idx = range_ms[idx]
                    da_idx = range_da[idx]
                    da_token = da_part.loc[da_idx].da_token
                    sent_id = da_part.loc[da_idx].sent_id
                    turn_id = da_part.loc[da_idx].turn_id
                    da_label = da_part.loc[da_idx].da_label
                    start_time = ms_part.loc[ms_idx].start_time
                    end_time = ms_part.loc[ms_idx].end_time
                    ms_token = ms_part.loc[ms_idx].word_norm
                    if not ms_token:
                        print(ms_part)
                    list_row.append({
                        'filenum': filenum,
                        'da_speaker': speaker,
                        'true_speaker': true_speaker,
                        'ms_token': ms_token,
                        'da_token': da_token,
                        'sent_id': sent_id,
                        'turn_id': turn_id,
                        'da_label': da_label,
                        'start_time': start_time,
                        'end_time': end_time
                        })
                if len_toks < len(range_da):
                    while idx < len(range_da)-1:
                        idx += 1
                        da_idx = range_da[idx]
                        da_token = da_part.loc[da_idx].da_token
                        sent_id = da_part.loc[da_idx].sent_id
                        turn_id = da_part.loc[da_idx].turn_id
                        da_label = da_part.loc[da_idx].da_label
                        ms_token = "<INSERTED>" 
                        # start/end time of ms stays the same 
                        if not ms_token:
                            print(ms_part)
                        list_row.append({
                            'filenum': filenum,
                            'da_speaker': speaker,
                            'true_speaker': true_speaker,
                            'ms_token': ms_token,
                            'da_token': da_token,
                            'sent_id': sent_id,
                            'turn_id': turn_id,
                            'da_label': da_label,
                            'start_time': start_time,
                            'end_time': end_time
                            })
                else: # len_toks > len(range_da)
                    while idx < len(range_ms)-1:
                        idx += 1
                        ms_idx = range_ms[idx]
                        da_token = "<MISSED>"
                        start_time = ms_part.loc[ms_idx].start_time
                        end_time = ms_part.loc[ms_idx].end_time
                        ms_token = ms_part.loc[ms_idx].word_norm
                        if not ms_token:
                            print(ms_part)
                        list_row.append({
                            'filenum': filenum,
                            'da_speaker': speaker,
                            'true_speaker': true_speaker,
                            'ms_token': ms_token,
                            'da_token': da_token,
                            'sent_id': sent_id,
                            'turn_id': turn_id,
                            'da_label': da_label,
                            'start_time': start_time,
                            'end_time': end_time
                            })
            else:
                #transfer token times from the back
                #print("back:", info, ms_side[i1:i2], da_side[j1:j2])
                temp_list = []
                for idx in range(len_toks):
                    ms_idx = i2 - idx - 1
                    da_idx = j2 - idx - 1
                    da_token = da_part.loc[da_idx].da_token
                    sent_id = da_part.loc[da_idx].sent_id
                    turn_id = da_part.loc[da_idx].turn_id
                    da_label = da_part.loc[da_idx].da_label
                    start_time = ms_part.loc[ms_idx].start_time
                    end_time = ms_part.loc[ms_idx].end_time
                    ms_token = ms_part.loc[ms_idx].word_norm
                    if not ms_token:
                        print(ms_part)
                    temp_list.append({
                        'filenum': filenum,
                        'da_speaker': speaker,
                        'true_speaker': true_speaker,
                        'ms_token': ms_token,
                        'da_token': da_token,
                        'sent_id': sent_id,
                        'turn_id': turn_id,
                        'da_label': da_label,
                        'start_time': start_time,
                        'end_time': end_time
                        })
                if len_toks < len(range_da):
                    while idx < len(range_da) - 1:
                        idx += 1
                        da_idx = range_da[len(range_da) - idx - 1]
                        da_token = da_part.loc[da_idx].da_token
                        sent_id = da_part.loc[da_idx].sent_id
                        turn_id = da_part.loc[da_idx].turn_id
                        da_label = da_part.loc[da_idx].da_label
                        ms_token = "<INSERTED>" 
                        # start/end time of ms stays the same 
                        if not ms_token:
                            print(ms_part)
                        temp_list.append({
                            'filenum': filenum,
                            'da_speaker': speaker,
                            'true_speaker': true_speaker,
                            'ms_token': ms_token,
                            'da_token': da_token,
                            'sent_id': sent_id,
                            'turn_id': turn_id,
                            'da_label': da_label,
                            'start_time': start_time,
                            'end_time': end_time
                            })
                else: # len_toks > len(range_da)
                    while idx < len(range_ms) - 1:
                        idx += 1
                        ms_idx = range_ms[len(range_ms) - idx - 1]
                        da_token = "<MISSED>"
                        start_time = ms_part.loc[ms_idx].start_time
                        end_time = ms_part.loc[ms_idx].end_time
                        ms_token = ms_part.loc[ms_idx].word_norm
                        if not ms_token:
                            print(ms_part)
                        temp_list.append({
                            'filenum': filenum,
                            'da_speaker': speaker,
                            'true_speaker': true_speaker,
                            'ms_token': ms_token,
                            'da_token': da_token,
                            'sent_id': sent_id,
                            'turn_id': turn_id,
                            'da_label': da_label,
                            'start_time': start_time,
                            'end_time': end_time
                            })
                list_row += temp_list[::-1]

    ret_df = pd.DataFrame(list_row)
    return ret_df
                

def get_alignments(config, split):
    da_dir = os.path.join(config.task_data_dir, split)
    list_df = []
    for filename in glob.glob(da_dir + '/*.tsv'):
        prefix = os.path.basename(filename)[2:4]
        filenum = os.path.basename(filename)[2:6]
        print(filenum)
        # debug         
        #if int(filenum) != 2005: continue
        speaker = os.path.basename(filename)[7]
        true_speaker = speaker
        sub_ms_dir = os.path.join(config.ms_dir, prefix, filenum)
        if filenum in SWITCHED:
            if speaker == 'A':
                true_speaker = 'B'
            else:
                true_speaker = 'A'
        ms_file = os.path.join(sub_ms_dir,
                'sw' + filenum + true_speaker + '-ms98-a-word.text')
        if filenum in SP6_FILES:
            if (filenum in SP6_NOTA and speaker == 'A') or \
                (filenum in SP6_NOTB and speaker == 'B'):
                sep = " "
            else:
                sep = "     "
        elif filenum in SP2_FILES:
            sep = "  "
        elif filenum in TAB_FILES:
            sep = "\t"
        else:
            sep = " "
        ms_df = pd.read_csv(ms_file, sep=sep, names=ms_cols)
        da_df = pd.read_csv(filename, sep='\t')
        da_unroll_df = unroll_toks(da_df)
        #print(filenum, speaker, true_speaker)
        df = align_dfs(da_unroll_df, ms_df, filenum, speaker, true_speaker)
        list_df.append(df)
    split_df = pd.concat(list_df)
    outname = os.path.join(config.out_dir, split + "_aligned.tsv")
    split_df.to_csv(outname, sep="\t", index=False)

def main():
    from config import SpeechConfig
    config = SpeechConfig()

    #get_align_checks(config, 'test')
    get_alignments(config, 'train')

    exit(0)


if __name__ == '__main__':
    main()
