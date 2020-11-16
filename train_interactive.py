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
from comet_ml import Experiment

# Create an experiment with your api key:
experiment = Experiment(
        api_key="YJEjlcG6ebIbBoOJ3sJizohqf",
        project_name="joint-seg-da",
        workspace="trangham283",
)

from swda_utils.config import TrainConfig as Config
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


lens_train = []
dev_data_source.epoch_init(shuffle=False)
while True:
    batch_data = dev_data_source.next(config.batch_size)
    if batch_data is None:
        break
    lens_train.append(batch_data['X'].size())

print("max_lens: ", max(lens_train), min(lens_train), np.mean(lens_train), np.median(lens_train))

#importlib.reload(sys.modules['model.joint_da_seg_recog.ctx_attn_ed'])
from model.joint_da_seg_recog.ctx_attn_ed import BertAttnEDSeqLabeler

model = BertAttnEDSeqLabeler(config, tokenizer, label_tokenizer, freeze='pooler_only')
model = BertAttnEDSeqLabeler(config, tokenizer, label_tokenizer, freeze='top_layer')
for name, param in model.named_parameters(): 
    if "bert.pooler." in name: print("\t", name, param.requires_grad)
    print(name, "size/grad: ", param.size(), param.requires_grad)


# model adaption
if torch.cuda.is_available():
    model = model.cuda()

if config.model_path:
    model.load_model(config.model_path)

# Build optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=config.init_lr,
    weight_decay=config.l2_penalty
)

# Build lr scheduler
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode="min",
    factor=config.lr_decay_rate,
    patience=2,
)

# log hyper parameters
start_time = time.time()
for k, v in sorted(dict(config.__dict__).items()):
    print("{}: {}".format(k, v))


# here we go
epoch = 1
n_step = 0

# Train
n_batch = 0

train_data_source.epoch_init(shuffle=False)

# GET NEW BATCH HERE
# one batch at a time
batch_data = train_data_source.next(config.batch_size)
if batch_data is None:
    print("End of data")

# Forward
model.train()
ret_data, ret_stat = model.train_step(batch_data)
trn_reporter.update_data(ret_stat)

# Backward
loss = ret_data["loss"]
loss.backward()
if config.gradient_clip > 0.0:
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config.gradient_clip
    )
optimizer.step()
optimizer.zero_grad()

# update
trn_reporter.update_data(ret_stat)

# Check loss
if n_step > 0 and n_step % config.check_loss_after_n_step == 0:
    log_s = f"{time.time()-start_time:.2f}s Epoch {epoch} batch {n_batch} - "
    log_s += trn_reporter.to_string()
    print(log_s)
    trn_reporter.clear()

# Evaluate on dev dataset
model.eval()

log_s = f"<Dev> learning rate: {lr}\n"
print(log_s)

pred_labels, true_labels = [], []
dev_data_source.epoch_init(shuffle=False)
while True:
    batch_data = dev_data_source.next(config.eval_batch_size)
    if batch_data is None:
        break
    ret_data, ret_stat = model.evaluate_step(batch_data)
    dev_reporter.update_data(ret_stat)
    ret_data, ret_stat = model.test_step(batch_data)
    refs = batch_data["Y"][:, 1:].tolist()
    hyps = ret_data["symbols"].tolist()
    for true_label_ids, pred_label_ids in zip(refs, hyps):
        end_idx = true_label_ids.index(label_tokenizer.eos_token_id)
        true_labels.append([label_tokenizer.id2word[label_id] for label_id in true_label_ids[:end_idx]])
        pred_labels.append([label_tokenizer.id2word[label_id] for label_id in pred_label_ids[:end_idx]])

log_s = f"\n<Dev> - {time.time()-start_time:.3f}s - "
log_s += dev_reporter.to_string()
print(log_s)
metrics_results = metrics.batch_metrics(true_labels, pred_labels)
log_s = \
    f"\tDSER:              {100*metrics_results['DSER']:.2f}\n" \
    f"\tseg WER:           {100*metrics_results['strict segmentation error']:.2f}\n" \
    f"\tDER:               {100*metrics_results['DER']:.2f}\n" \
    f"\tjoint WER:         {100*metrics_results['strict joint error']:.2f}\n" \
    f"\tMacro F1:          {100*metrics_results['Macro F1']:.2f}\n" \
    f"\tMicro F1:          {100*metrics_results['Micro F1']:.2f}\n" \
    f"\tMacro LWER:        {100*metrics_results['Macro LWER']:.2f}\n" \
    f"\tMicro LWER:        {100*metrics_results['Micro LWER']:.2f}\n"
print(log_s)

# Decay learning rate
lr_scheduler.step(dev_reporter.get_value("monitor"))
dev_reporter.clear()

# Finished a step
n_batch += 1
n_step += 1

experiment.log_metrics(metrics_results)
