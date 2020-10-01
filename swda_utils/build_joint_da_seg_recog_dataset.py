import code
import argparse
import json
import os
import random
import re
import collections
from multiprocessing import Pool
from tqdm import tqdm
import spacy
import string
import sys
import pandas as pd

from swda_reader.swda import Transcript
from swda_reader.swda import CorpusReader

# default config file:
# config_file = "/homes/ttmt001/transitory/dialog-act-prediction/joint_seg_da/swda_utils/config.py")

from config import Config
config = Config()
with open(config.train_dialog_list) as fin:
    train_set_idx = set([
        '{0}'.format(line.strip())
        for line in fin
    ])
with open(config.test_dialog_list) as fin:
    test_set_idx = set([
        '{0}'.format(line.strip())
        for line in fin
    ])

nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])

# Nonverbals can be in the form of <throat_clearing> or throat_clearing.
# nonverbal_reg_exp = r"\<+[^\<\>]+\>+|[a-zA-Z]+_[a-zA-Z]+"
nonverbal_reg_exp_list = [re.compile(r"\<+[^\<\>]+\>+"), re.compile(r"\w+_\w+")]
nonverbal_token = "NONVERBAL"
skip_punt_reg_exp = re.compile(r"[#\-\(\)]+")

def clean_swda_text(utterance, lowercase=True, skip_punct=True,
        filter_disfluency=True, skip_mark_nonverbal=False):
    utterance_text = ' '.join(
        utterance.text_words(filter_disfluency=filter_disfluency)
    )
    for nonverbal_reg_exp in nonverbal_reg_exp_list:
        find_match = re.search(nonverbal_reg_exp, utterance_text)
        if skip_mark_nonverbal:
            # Removes the non-verbal expression.
            utterance_text = re.sub(nonverbal_reg_exp, ' ', utterance_text)
        else:
            # Normalizes the non-verbal expression.
            utterance_text = re.sub(nonverbal_reg_exp, nonverbal_token,
                    utterance_text)
    utterance_text = re.sub(skip_punt_reg_exp, ' ', utterance_text)
    doc = nlp(utterance_text)
    word_list = [token.text.lower()
        if lowercase and token.text != nonverbal_token else token.text
        for token in doc]
    if skip_punct:
        # Skips punctuation if required.
        word_list = [word.strip() for word in word_list if word not in string.punctuation]
        word_list = [word for word in word_list if word != "..."]
        word_list = [word for word in word_list if word]
    return word_list


def build_session(filepath):
    trans = Transcript(filepath, f"{config.raw_data_dir}/swda-metadata.csv")
    conversation_no = f"sw{trans.conversation_no}"
    topic = trans.topic_description.lower()
    prompt = trans.prompt.lower()
    # adapt utterances with "+" dialog act
    segments = []
    last_idx = {"A": -1, "B": -1}
    for uttr in trans.utterances:
        word_list = clean_swda_text(uttr)
        text = " ".join(word_list)
        da = uttr.damsl_act_tag()
        speaker = uttr.caller
        segment_idx = uttr.transcript_index
        turn_idx = uttr.utterance_index
        if not word_list or da is None:
            print('Empty utterance: ', file=sys.stdout)
            print('\t{} {} {} {}'.format(uttr.swda_filename, segment_idx, turn_idx, da), file=sys.stdout)
            continue
        elif da == "x":
            continue
        elif da == "+":
            if last_idx[speaker] > -1:
                segments[last_idx[speaker]]["text"] += f" {text}"
            else:
                continue
        else:
            segment = {"floor": speaker,
                "text": text, 
                "segment_meta": {
                    "dialog_act": da,
                    "turn_idx": turn_idx,
                    "segment_idx": segment_idx
                }
            }
            segments.append(segment)
            last_idx[speaker] = len(segments)-1
        if da == "" or text.strip() == "":
            print("empty text: ")
            print('\t', uttr.swda_filename, segment_idx, turn_idx, da)
            #code.interact(local=locals())
    # get turns from segments
    uttrs = []
    uttr = []
    last_speaker = ""
    for segment in segments:
        speaker = segment["floor"]
        if speaker == last_speaker:
            uttr.append(segment)
        else:
            if len(uttr) > 0:
                uttrs.append(uttr)
            uttr = [segment]
            last_speaker = speaker
    uttrs.append(uttr)

    #NOTE on terminology: "utterances" here means "turns"
    session = {
        "utterances": uttrs,
        "dialog_meta": {
            "conversation_no": conversation_no,
            "topic": topic,
            "prompt": prompt,
        }
    }
    
    if conversation_no in train_set_idx:
        split = 'train'
    elif conversation_no in test_set_idx:
        split = 'test'
    else:
        split = 'dev'

    outdir = os.path.join(config.task_data_dir, split)
    a_df, b_df = build_tsv(session)
    a_df.to_csv(outdir + "/"+conversation_no + '_A.tsv', sep="\t", index=False)
    b_df.to_csv(outdir + "/"+conversation_no + '_B.tsv', sep="\t", index=False)
    return session

def train_dev_test_split_by_conv_no(sessions):
    dataset = {"train": [], "dev": [], "test": []}
    for session in sessions:
        conv_no = session["dialog_meta"]["conversation_no"]
        if conv_no in train_set_idx:
            dataset["train"].append(session)
        elif conv_no in test_set_idx:
            dataset["test"].append(session)
        else:
            dataset["dev"].append(session)
    return dataset

def build_tsv(sess):
    a_tsv = []
    b_tsv = []
    filename = sess['dialog_meta']['conversation_no']
    for turn in sess['utterances']:
        for segment in turn:
            speaker = segment['floor']
            sentence_id = segment['segment_meta']['segment_idx']
            sentence_id = f"{filename}_{speaker}_{str(sentence_id).zfill(4)}" 
            if speaker == 'A':
                a_tsv.append({
                    'normed_text': segment['text'],
                    'speaker': speaker,
                    'da_label': segment['segment_meta']['dialog_act'],
                    'turn_id': segment['segment_meta']['turn_idx'],
                    'sentence_id': sentence_id
                    })
            else:
                b_tsv.append({
                    'normed_text': segment['text'],
                    'speaker': speaker,
                    'da_label': segment['segment_meta']['dialog_act'],
                    'turn_id': segment['segment_meta']['turn_idx'],
                    'sentence_id': sentence_id
                    })
    a_df = pd.DataFrame(a_tsv)
    b_df = pd.DataFrame(b_tsv)
    return a_df, b_df



if __name__ == "__main__":
    corpus = CorpusReader(config.raw_data_dir)
    
    sessions = []
    print("Extracting sessions...")
    # Get file paths
    filepaths = []
    for trans in corpus.iter_transcripts():
        filepaths.append(trans.swda_filename)
    with Pool(config.n_workers) as pool:
        sessions = list(
            tqdm(
                pool.imap(build_session, filepaths),
                total=len(filepaths)
            )
        )

    print("Building dataset split...")
    dataset = train_dev_test_split_by_conv_no(sessions)

    print(f"Dialog act tags:")
    da_dict = collections.defaultdict(int)
    for sess in dataset["train"]:
        for uttr in sess["utterances"]:
            for segment in uttr:
                da = segment["segment_meta"]["dialog_act"]
                da_dict[da] += 1
    sorted_da_list = sorted(da_dict.items(), key=lambda x: x[1], reverse=True)
    print(len(sorted_da_list))
    print([k for k, v in sorted_da_list])


    print(f"Writing dataset json file to {config.dataset_path}...")
    if not os.path.exists(config.task_data_dir):
        os.makedirs(config.task_data_dir)

    with open(config.dataset_path, "w+", encoding="utf-8") as f:
        json.dump(dataset, f)
    print("Dataset built.")

    print(f"Building word count file...")
    word_count = collections.defaultdict(int)
    for sess in dataset["train"]:
        for uttr in sess["utterances"]:
            for segment in uttr:
                for token in segment["text"].split():
                    word_count[token] += 1
    ordered_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    with open(config.word_count_path, "w+") as f:
        for word, count in ordered_word_count:
            f.write("{}\t{}\n".format(word, count))

    

