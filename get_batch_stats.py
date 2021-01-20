import json
import random
import code
import time
import sys
import math
import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import importlib

from utils.helpers import StatisticsReporter
from utils.metrics import JointDAMetrics as DAMetrics
from tokenization.bert_tokenizer import ModBertTokenizer
from tokenization.customized_tokenizer import CustomizedTokenizer
from data_source import SpeechDataSource

from swda_utils.config import SpeechConfig as Config
config = Config()
config.seed = 42

# set random seeds
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)

tokenizer = ModBertTokenizer('base', cache_dir="/s0/ttmt001")

label_token_dict = {
    f"label_{label_idx}_token": label 
    for label_idx, label in enumerate(config.joint_da_seg_recog_labels)
}
label_token_dict.update({
    "pad_token": "<pad>",
    "bos_token": "<t>",
    "eos_token": "</t>"
})
label_tokenizer = CustomizedTokenizer(
    token_dict=label_token_dict
)

# data loaders & number reporters
#split = 'test'

#config.suffix = "_bert_time_data.json"
#data_source = SpeechDataSource(split=split, config=config, 
#        tokenizer=tokenizer, label_tokenizer=label_tokenizer)
#
#lenstats = []
#for dialog, fragments in data_source.fragments.items():
#    for frag in fragments:
#        turn = frag[-1]
#        lentok = len(turn['token_ids'])
#        lenstats.append(lentok)
#
#
#print(min(lenstats), max(lenstats), np.mean(lenstats), np.median(lenstats))
#print(len([x for x in lenstats if x > 45]))
#print(len([x for x in lenstats if x > 100]))

#################################################
#config.suffix = "_bert_time_data.json"
#
#split = 'train'
#
#config.feature_types = []
#config.history_len = 1
#config.fixed_word_length = 50
#config.downsample = True
#data_source = SpeechDataSource(split=split, config=config, 
#        tokenizer=tokenizer, label_tokenizer=label_tokenizer)
#
#batch_size = 32
#batch_lens = []
#
#dialog_keys = data_source.dialog_keys
#for dialog_idx in dialog_keys:
#    dialog_length = data_source.get_dialog_length(dialog_idx)
#    turn_keys = list(range(dialog_length))
#    dialog_frames = []
#    random.shuffle(turn_keys)
#    for offset in range(0, dialog_length, batch_size):
#        turn_idx = turn_keys[offset:offset+batch_size]
#        batch_data = data_source.get_batch_features(dialog_idx, 
#                dialog_frames, turn_idx)
#        seq_len = batch_data["X"].size(-1)
#        batch_lens.append(seq_len)
#
#print(min(batch_lens), max(batch_lens), np.mean(batch_lens), np.median(batch_lens))



#################################################

split = 'dev'
fname = split + "_bert_time_data.json"

with open(fname, 'r') as f:
    data = json.load(f)

num_turns = []
num_utts = []
for k, dialog in data.items():
    num_turns.append(len(dialog)
    for turn in dialog:
        num_utts.append(len(turn['sent_ids']))




