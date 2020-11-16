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
from data_source import TextDataSource

from swda_utils.config import BertTrainConfig as Config
config = Config()


# merge parse args with corpus config
# priority: parse args > corpus config
config.model = 'attn_ed'
config.rnn_type = 'gru'
config.tie_weights = True
config.attention_type = 'sent'

# model - numbers
config.vocab_size = 30000
config.history_len = 3
config.word_embedding_dim = 300 # default for glove
config.attr_embedding_dim = 30
config.sent_encoder_hidden_dim = 100
config.n_sent_encoder_layers = 1
config.dial_encoder_hidden_dim = 60
config.n_dial_encoder_layers = 1
config.decoder_hidden_dim = 600
config.n_decoder_layers = 1

# training
config.seed = 0
config.max_uttr_len = 40
config.dropout = 0.2
config.l2_penalty = 0.0001
config.optimizer = 'adam'
config.init_lr = 0.001
config.min_lr = 1e-7
config.lr_decay_rate = 0.5
config.gradient_clip = 5.0
config.n_epochs = 10
config.use_pretrained_word_embedding = True
config.batch_size = 4
config.eval_batch_size = 4

# inference
config.decode_max_len = 40
config.gen_type = 'greedy' # "[greedy, sample, top]"
config.temp = 1.0
config.top_k = 0
config.top_p = 0.0

# paths
# config.model_path = "/s0/ttmt001/da_debugs"
config.model_path = None
config.corpus = 'swda'
config.enable_log = False
config.save_model = False
config.check_loss_after_n_step = 100
config.validate_after_n_step = 100
config.filename_note = 'debug'


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
trn_reporter = StatisticsReporter()
dev_reporter = StatisticsReporter()
with open(config.dataset_path, encoding="utf-8") as f:
    dataset = json.load(f)


train_data_source = TextDataSource(
    data=dataset["train"],
    config=config,
    tokenizer=tokenizer,
    label_tokenizer=label_tokenizer
)

dev_data_source = TextDataSource(
    data=dataset["dev"],
    config=config,
    tokenizer=tokenizer,
    label_tokenizer=label_tokenizer
)


data = dataset['test']
lens_bert = []
lens_ws = []
diffs = []
for sess in data:
    for uttr in sess["utterances"]:
        uttr_tokens = []
        for segment in uttr:
            text = segment["text"]
            tokens_bert = tokenizer.convert_string_to_tokens(text)
            tokens_ws = text.split() 
            lens_bert.append(len(tokens_bert))
            lens_ws.append(len(tokens_ws))
            diffs.append(len(tokens_bert) - len(tokens_ws))


thres = 45
print("stats: ", len(lens_bert), max(lens_bert), min(lens_bert), np.mean(lens_bert), np.median(lens_bert), len([x for x in lens_bert if x > thres]))
print("stats: ", len(lens_ws), max(lens_ws), min(lens_ws), np.mean(lens_ws), np.median(lens_ws), len([x for x in lens_ws if x > thres]))
print("stats: ", len(diffs), max(diffs), min(diffs), np.mean(diffs), np.median(diffs))

#dev_data_source.epoch_init(shuffle=False)
#while True:
#    batch_data = dev_data_source.next(config.batch_size)
#    if batch_data is None:
#        break
#    lens_train.append(batch_data['X'].size())
#
#print("max_lens: ", max(lens_train), min(lens_train), np.mean(lens_train), np.median(lens_train))




#importlib.reload(sys.modules['model.joint_da_seg_recog.ctx_attn_ed'])
#from model.joint_da_seg_recog.ctx_attn_ed import BertAttnEDSeqLabeler

#damodel = BertAttnEDSeqLabeler(config, tokenizer, label_tokenizer)
#print(damodel)



