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

from model.joint_da_seg_recog.ed import EDSeqLabeler
from model.joint_da_seg_recog.attn_ed import AttnEDSeqLabeler
from utils.helpers import StatisticsReporter
from utils.metrics import DAMetrics
from tokenization.whitespace_tokenizer import WhiteSpaceTokenizer
from tokenization.customized_tokenizer import CustomizedTokenizer
from data_source import DataSource

from swda_utils.config import Config
config = Config()

# merge parse args with corpus config
# priority: parse args > corpus config
config.model = 'attn_ed'
config.rnn_type = 'lstm'
config.tie_weights = True
config.attention_type = 'word'

# model - numbers
config.vocab_size = 10000
config.history_len = 3
config.word_embedding_dim = 300 # default for glove
config.attr_embedding_dim = 30
config.sent_encoder_hidden_dim = 100
config.n_sent_encoder_layers = 1
config.dial_encoder_hidden_dim = 200
config.n_dial_encoder_layers = 1
config.decoder_hidden_dim = 200
config.n_decoder_layers = 1

# training
config.seed = 42
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
config.batch_size = 30
config.eval_batch_size = 60

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

def mlog(s, LOG_FILE_NAME):
    if config.enable_log:
        if not os.path.exists(f"{config.model_path}"):
            os.makedirs(f"{config.model_path}")
        with open(f"{config.model_path}/{LOG_FILE_NAME}.log", "a+", encoding="utf-8") as log_f:
            log_f.write(s+"\n")
    print(s)


# define logger
MODEL_NAME = config.model
LOG_FILE_NAME = "{}.seed_{}.{}".format(
    MODEL_NAME,
    config.seed,
    time.strftime("%Y%m%d-%H%M%S", time.localtime())
)
if config.filename_note:
    LOG_FILE_NAME += f".{config.filename_note}"

# set random seeds
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
random.seed(config.seed)
np.random.seed(config.seed)

# tokenizers
special_token_dict = {
    "speaker1_token": "<speaker1>",
    "speaker2_token": "<speaker2>"
}
tokenizer = WhiteSpaceTokenizer(
    word_count_path=config.word_count_path,
    vocab_size=config.vocab_size,
    special_token_dict=special_token_dict
)
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


mlog("----- Loading training data -----", LOG_FILE_NAME)
train_data_source = DataSource(
    data=dataset["train"],
    config=config,
    tokenizer=tokenizer,
    label_tokenizer=label_tokenizer
)
mlog(str(train_data_source.statistics), LOG_FILE_NAME)


mlog("----- Loading dev data -----", LOG_FILE_NAME)
dev_data_source = DataSource(
    data=dataset["dev"],
    config=config,
    tokenizer=tokenizer,
    label_tokenizer=label_tokenizer
)
mlog(str(dev_data_source.statistics), LOG_FILE_NAME)


mlog("----- Loading test data -----", LOG_FILE_NAME)
test_data_source = DataSource(
    data=dataset["test"],
    config=config,
    tokenizer=tokenizer,
    label_tokenizer=label_tokenizer
)
mlog(str(test_data_source.statistics), LOG_FILE_NAME)

# metrics calculator
metrics = DAMetrics()

# build model
if config.model == "ed":
    Model = EDSeqLabeler
elif config.model == "attn_ed":
    Model = AttnEDSeqLabeler

model = Model(config, tokenizer, label_tokenizer)

# model adaption
if torch.cuda.is_available():
    mlog("----- Using GPU -----", LOG_FILE_NAME)
    model = model.cuda()

if config.model_path:
    model.load_model(config.model_path)
    mlog("----- Model loaded -----", LOG_FILE_NAME)
    mlog(f"model path: {config.model_path}", LOG_FILE_NAME)

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
mlog("----- Hyper-parameters -----", LOG_FILE_NAME)
for k, v in sorted(dict(config.__dict__).items()):
    mlog("{}: {}".format(k, v), LOG_FILE_NAME)

# here we go
n_step = 0
for epoch in range(1, config.n_epochs+1):
    lr = list(lr_scheduler.optimizer.param_groups)[0]["lr"]
    if lr <= config.min_lr:
        break

    # Train
    n_batch = 0
    train_data_source.epoch_init(shuffle=True)
    while True:
        batch_data = train_data_source.next(config.batch_size)
        if batch_data is None:
            break

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
            mlog(log_s)
            trn_reporter.clear()

        # Evaluate on dev dataset
        if n_step > 0 and n_step % config.validate_after_n_step == 0:
            model.eval()

            log_s = f"<Dev> learning rate: {lr}\n"
            mlog(log_s)

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
            mlog(log_s)
            metrics_results = metrics.batch_metrics(true_labels, pred_labels)
            log_s = \
                f"\tDSER:            {100*metrics_results['DSER']:.2f}\n" \
                f"\tseg WER:         {100*metrics_results['strict segmentation error']:.2f}\n" \
                f"\tDER:             {100*metrics_results['DER']:.2f}\n" \
                f"\tjoint WER:       {100*metrics_results['strict joint error']:.2f}\n" \
                f"\tMacro F1:        {100*metrics_results['Macro F1']:.2f}\n" \
                f"\tMicro F1:        {100*metrics_results['Micro F1']:.2f}\n"
            mlog(log_s)

            # Save model if it has better monitor measurement
            if config.save_model:
                if not os.path.exists(f"../data/{config.corpus}/model/{config.task}"):
                    os.makedirs(f"../data/{config.corpus}/model/{config.task}")

                torch.save(model.state_dict(), f"../data/{config.corpus}/model/{config.task}/{LOG_FILE_NAME}.model.pt")
                mlog(f"model saved to data/{config.corpus}/model/{config.task}/{LOG_FILE_NAME}.model.pt")

                if torch.cuda.is_available():
                    model = model.cuda()

            # Decay learning rate
            lr_scheduler.step(dev_reporter.get_value("monitor"))
            dev_reporter.clear()

        # Finished a step
        n_batch += 1
        n_step += 1
    
    # Evaluate on test dataset every epoch
    model.eval()
    pred_labels, true_labels = [], []
    test_data_source.epoch_init(shuffle=False)
    while True:
        batch_data = test_data_source.next(config.eval_batch_size)
        if batch_data is None:
            break

        ret_data, ret_stat = model.test_step(batch_data)
        
        refs = batch_data["Y"][:, 1:].tolist()
        hyps = ret_data["symbols"].tolist()
        for true_label_ids, pred_label_ids in zip(refs, hyps):
            end_idx = true_label_ids.index(label_tokenizer.eos_token_id)
            true_labels.append([label_tokenizer.id2word[label_id] for label_id in true_label_ids[:end_idx]])
            pred_labels.append([label_tokenizer.id2word[label_id] for label_id in pred_label_ids[:end_idx]])

    log_s = f"\n<Test> - {time.time()-start_time:.3f}s - "
    mlog(log_s)
    metrics_results = metrics.batch_metrics(true_labels, pred_labels)
    log_s = \
        f"\tDSER:            {100*metrics_results['DSER']:.2f}\n" \
        f"\tseg WER:         {100*metrics_results['strict segmentation error']:.2f}\n" \
        f"\tDER:             {100*metrics_results['DER']:.2f}\n" \
        f"\tjoint WER:       {100*metrics_results['strict joint error']:.2f}\n" \
        f"\tMacro F1:        {100*metrics_results['Macro F1']:.2f}\n" \
        f"\tMicro F1:        {100*metrics_results['Micro F1']:.2f}\n"
    mlog(log_s)
