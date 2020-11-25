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
config.downsample = False
config.feature_types = ['pitch', 'fb3', 'pause', 'pause_raw', 'word_dur']
config.suffix = "_bert_time_data.json"

split = 'test'
data_source = SpeechDataSource(split=split, config=config, 
        tokenizer=tokenizer, label_tokenizer=label_tokenizer)

lenstats = []
for dialog, fragments in data_source.fragments.items():
    for frag in fragments:
        turn = frag[-1]
        lentok = len(turn['token_ids'])
        lenstats.append(lentok)


print(min(lenstats), max(lenstats), np.mean(lenstats), np.median(lenstats))
print(len([x for x in lenstats if x > 45]))
print(len([x for x in lenstats if x > 100]))



