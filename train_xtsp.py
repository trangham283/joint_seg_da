#!/usr/bin/env python

import json
import random
import code
import time
import sys
import math
import argparse
import os

from comet_ml import Experiment

import numpy as np
import torch
import torch.optim as optim
import transformers

from utils.helpers import StatisticsReporter
from utils.metrics import JointDAMetrics as DAMetrics
from tokenization.customized_tokenizer import CustomizedTokenizer
from tokenization.bert_tokenizer import ModBertTokenizer
from model.joint_da_seg_recog.ctx_attn_ed import SpeechTransformerLabeler
from data_source import SpeechXTSource

# Create an experiment with your api key:
experiment = Experiment(
        api_key="YJEjlcG6ebIbBoOJ3sJizohqf",
        project_name="joint-seg-da",
        workspace="trangham283",
        )

def str2bool(v):
    return v.lower() in ('true', '1', "True")

def mlog(s, config, LOG_FILE_NAME):
    if config.enable_log:
        if not os.path.exists(f"{config.task_data_dir}/log/"):
            os.makedirs(f"{config.task_data_dir}/log/")

        with open(f"{config.task_data_dir}/log/{LOG_FILE_NAME}.log", "a+", 
                encoding="utf-8") as log_f:
            log_f.write(s+"\n")
    print(s)

def reslog(s, RES_FILE_NAME):
    if not os.path.exists(f"{config.task_data_dir}/log/"):
        os.makedirs(f"{config.task_data_dir}/log/")

    with open(f"{config.task_data_dir}/log/{RES_FILE_NAME}.res", "a+", 
            encoding="utf-8") as log_f:
        log_f.write(s+"\n")

def eval_split(model, data_source, set_name, config, label_tokenizer, 
        metrics, LOG_FILE_NAME, write_pred=False):
    if write_pred:
        RES_FILE_NAME = set_name + "_" + LOG_FILE_NAME
        s = "LABELS\tPREDS"
        reslog(s, RES_FILE_NAME)

    pred_labels, true_labels = [], []
    total_loss = 0
    for dialog_idx in data_source.dialog_keys:
        if config.frame_features:
            dialog_frames = data_source.load_frames(dialog_idx)
        else:
            dialog_frames = []
        dialog_length = data_source.get_dialog_length(dialog_idx)
        turn_keys = list(range(dialog_length))
        for offset in range(0, dialog_length, config.eval_batch_size):
            turn_idx = turn_keys[offset:offset+config.eval_batch_size]
            batch_data = data_source.get_batch_features(dialog_idx, 
                    dialog_frames, turn_idx)
            
            # Forward
            ret_data, ret_stat = model.test_step(batch_data)
            batch_loss = ret_data["batch_loss"]
            if batch_loss is not None:
                total_loss += batch_loss
            
            refs = batch_data["Y"][:, 1:].tolist()
            hyps = ret_data["symbols"].squeeze(-1).tolist()
            for true_label_ids, pred_label_ids in zip(refs, hyps):
                end_idx = true_label_ids.index(label_tokenizer.eos_token_id)
                true_syms = [label_tokenizer.id2word[label_id] 
                        for label_id in true_label_ids[:end_idx]]
                pred_syms = [label_tokenizer.id2word[label_id] 
                        for label_id in pred_label_ids[:end_idx]]
                if write_pred:
                    s = " ".join(true_syms) + "\t" + " ".join(pred_syms) 
                    reslog(s, RES_FILE_NAME)
                true_labels.append(true_syms)
                pred_labels.append(pred_syms)

    log_s = f"\nSplit: {set_name} - Results - "
    mlog(log_s, config, LOG_FILE_NAME)
    metrics_results = metrics.batch_metrics(true_labels, pred_labels)
    log_s = \
        f"\tDSER:            {100*metrics_results['DSER']:.2f}\n" \
        f"\tseg WER:         {100*metrics_results['strict segmentation error']:.2f}\n" \
        f"\tDER:             {100*metrics_results['DER']:.2f}\n" \
        f"\tjoint WER:       {100*metrics_results['strict joint error']:.2f}\n" \
        f"\tMacro F1:        {100*metrics_results['Macro F1']:.2f}\n" \
        f"\tMicro F1:        {100*metrics_results['Micro F1']:.2f}\n" \
        f"\tMacro LWER:      {100*metrics_results['Macro LWER']:.2f}\n" \
        f"\tMicro LWER:      {100*metrics_results['Micro LWER']:.2f}\n"
    mlog(log_s, config, LOG_FILE_NAME)
    current_score = -metrics_results['DER'] + metrics_results['Macro F1']
    if batch_loss is not None:
        split_loss = total_loss / data_source.statistics['n_turns']
    else:
        split_loss = None
    return current_score, metrics_results, split_loss


def run_train(config):
    # tokenizers
    tokenizer = ModBertTokenizer('base', cache_dir=config.cache_dir)
    label_token_dict = {
            "pad_token": "<pad>",
            "bos_token": "<t>",
            "eos_token": "</t>",}
    label_token_dict.update({
        f"label_{label_idx}_token": label 
        for label_idx, label in enumerate(config.joint_da_seg_recog_labels)
    })
    label_tokenizer = CustomizedTokenizer(
        token_dict=label_token_dict
    )

    # metrics calculator
    metrics = DAMetrics()

    # define logger
    MODEL_NAME = config.model
    LOG_FILE_NAME = "{}.seed_{}.{}".format(
        MODEL_NAME,
        config.seed,
        time.strftime("%Y%m%d-%H%M%S", time.localtime())[-6:]
    )
    if config.filename_note:
        LOG_FILE_NAME += f".{config.filename_note}"
        experiment.set_name(config.filename_note)
    experiment.log_text(LOG_FILE_NAME)

    # data loaders & number reporters
    trn_reporter = StatisticsReporter()
    mlog("----- Loading dev data -----", config, LOG_FILE_NAME)
    dev_data_source = SpeechXTSource(
        split="dev", 
        config=config,
        tokenizer=tokenizer,
        label_tokenizer=label_tokenizer
    )
    mlog(str(dev_data_source.statistics), config, LOG_FILE_NAME)

    mlog("----- Loading training data -----", config, LOG_FILE_NAME)
    if config.debug:
        train_data_source = dev_data_source
    else:
        train_data_source = SpeechXTSource(
            split="train", 
            config=config,
            tokenizer=tokenizer,
            label_tokenizer=label_tokenizer)
    mlog(str(train_data_source.statistics), config, LOG_FILE_NAME)
    

    # build model
    model = SpeechTransformerLabeler(config, tokenizer, label_tokenizer, freeze=config.freeze)

    # model adaption
    if torch.cuda.is_available():
        mlog("----- Using GPU -----", config, LOG_FILE_NAME)
        model = model.cuda()
    if config.model_path:
        model.load_model(config.model_path)
        mlog("----- Model loaded -----", config, LOG_FILE_NAME)
        mlog(f"model path: {config.model_path}", config, LOG_FILE_NAME)

    this_model_path = f"{config.model_save_path}/model"

    # Build optimizer
    trainable_parameters = [param for param in model.named_parameters() 
            if param[1].requires_grad]
    total_params_count = sum([x[1].numel() for x in trainable_parameters])
    print("Total params count: ", total_params_count)

    warmup_steps = math.ceil(train_data_source.statistics['n_turns'] * config.n_epochs / config.batch_size * 0.1) #10% of train data for warm-up
    # additional steps because of different loading schemes
    t_total = math.ceil(1.5 * train_data_source.statistics['n_turns'] * config.n_epochs / config.batch_size)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in trainable_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': config.lr_decay_rate},
        {'params': [p for n, p in trainable_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = transformers.AdamW(
        optimizer_grouped_parameters,
        lr=config.init_lr,
        weight_decay=config.lr_decay_rate,
        correct_bias=False,
    )

    # Build lr scheduler
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer=optimizer,
    #    mode="min",
    #    factor=config.lr_decay_rate,
    #    patience=2,
    #)
    print("warmup/total steps:", warmup_steps, t_total)
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=t_total) 


    # log hyper parameters
    start_time = time.time()
    mlog("----- Hyper-parameters -----", config, LOG_FILE_NAME)
    for k, v in sorted(dict(config.__dict__).items()):
        mlog("{}: {}".format(k, v), config, LOG_FILE_NAME)
    for name, param in model.named_parameters():
        mlog("{}: {}; Grad: {}".format(name, param.size(), param.requires_grad), config, LOG_FILE_NAME)

    
    # data stats for batching
    train_dialogs_keys = train_data_source.dialog_keys
    shuffle_dialogs = train_dialogs_keys[:]

    # TRAIN
    n_step = 0
    best_score = -9999
    best_loss = np.inf
    for epoch in range(1, config.n_epochs+1):
        #lr = list(lr_scheduler.optimizer.param_groups)[0]["lr"]
        #if lr <= config.min_lr:
        #    break
        lr = lr_scheduler.get_last_lr()

        random.shuffle(shuffle_dialogs)
        n_batch = 0
        for dialog_idx in shuffle_dialogs:
            if config.frame_features:
                dialog_frames = train_data_source.load_frames(dialog_idx)
            else:
                dialog_frames = []
            dialog_length = train_data_source.get_dialog_length(dialog_idx)
            turn_keys = list(range(dialog_length))
            random.shuffle(turn_keys)

            #if config.debug and n_step > 30: 
            #    break
            
            for offset in range(0, dialog_length, config.batch_size):
                model.zero_grad()
                model.train()
                turn_idx = turn_keys[offset:offset+config.batch_size]
                batch_data = train_data_source.get_batch_features(dialog_idx, 
                        dialog_frames, turn_idx)
                
                # Forward
                ret_data, ret_stat = model.train_step(batch_data)

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
                lr_scheduler.step()

                # update
                trn_reporter.update_data(ret_stat)

                # Check loss and Evaluate on dev dataset
                # Check loss
                if n_step > 0 and n_step % config.check_loss_after_n_step == 0:
                    log_s = f"{time.time()-start_time:.2f}s Epoch {epoch} batch {n_batch} step {n_step} - Training loss on this batch: "
                    log_s += trn_reporter.to_string()
                    mlog(log_s, config, LOG_FILE_NAME)
                    trn_reporter.clear()

                # evaluate 
                if n_step > 0 and n_step % config.validate_after_n_step == 0:
                    model.eval()

                    log_s = f"<Dev> learning rate: {lr}\n"
                    mlog(log_s, config, LOG_FILE_NAME)

                    current_score, metrics_results, split_loss \
                            = eval_split(model, dev_data_source, "dev", 
                            config, label_tokenizer, metrics, 
                            LOG_FILE_NAME, write_pred=False)
                    print("Split loss & best loss ", split_loss, best_loss)
                    print("Split score & best score ", current_score, best_score)
                    if not config.debug:
                        experiment.log_metrics(metrics_results)

                    if current_score > best_score:
                        best_score = current_score
                    # Save model if it has better monitor measurement
                    if split_loss < best_loss:
                        best_loss = split_loss
                        if config.save_model:
                            this_model_path = f"{config.model_save_path}/model"
                            if not os.path.exists(this_model_path):
                                os.makedirs(this_model_path)

                            torch.save(model.state_dict(), f"{this_model_path}/{LOG_FILE_NAME}.model.pt")
                            torch.save(config, f"{this_model_path}/{LOG_FILE_NAME}.config")
                            mlog(f"model saved to {this_model_path}/{LOG_FILE_NAME}.model.pt", config, LOG_FILE_NAME)

                # Finished a step
                n_batch += 1
                n_step += 1

        # Decay learning rate at end of epoch
        mlog("----- EVALUATING at end of epoch -----", config, LOG_FILE_NAME)
        mlog(f"End of epoch: {epoch}", config, LOG_FILE_NAME)
        current_score, metrics_results, split_loss = eval_split(
                model, dev_data_source, "dev", 
                config, label_tokenizer, metrics, 
                LOG_FILE_NAME, write_pred=False)
        print("Split loss & best loss ", split_loss, best_loss)
        print("Split score & best score ", current_score, best_score)
        if current_score > best_score:
            best_score = current_score
        if split_loss < best_loss:
            best_loss = split_loss
            if config.save_model:
                torch.save(model.state_dict(), f"{this_model_path}/{LOG_FILE_NAME}.model.pt")
                torch.save(config, f"{this_model_path}/{LOG_FILE_NAME}.config")
                mlog(f"model saved to {this_model_path}/{LOG_FILE_NAME}.model.pt", config, LOG_FILE_NAME)
        if not config.debug:
            experiment.log_metrics(metrics_results)
        #lr_scheduler.step(best_loss)

        
    # Evaluate on test dataset at the end of training
    mlog("----- EVALUATING at end of training -----", config, LOG_FILE_NAME)
    mlog("----- Loading test data -----", config, LOG_FILE_NAME)
    test_data_source = SpeechXTSource(
        split='test',
        config=config,
        tokenizer=tokenizer,
        label_tokenizer=label_tokenizer
    )
    mlog(str(test_data_source.statistics), config, LOG_FILE_NAME)
    if config.save_model:
        model_path = f"{this_model_path}/{LOG_FILE_NAME}.model.pt"
        model.load_model(model_path)
        print(f"model path: {model_path}")
    model.eval()

    for set_name, data_source in [("DEV", dev_data_source), ("TEST", test_data_source)]:
        current_score, metrics_results, split_loss = eval_split(
                model, data_source, set_name, 
                config, label_tokenizer, metrics, 
                LOG_FILE_NAME, write_pred=True)
        print("Split loss: ", split_loss)
        diff = (metrics_results['Macro F1'] - metrics_results['DER']) * 100

        lazy_s = f"DSER, DER, F1, LWER:\n {100*metrics_results['DSER']}\t{100*metrics_results['DER']}\t{100*metrics_results['Macro F1']}\t{diff}\t{100*metrics_results['Macro LWER']}\n"
        mlog(lazy_s, config, LOG_FILE_NAME)

# TODO
def run_test(config):
    # tokenizers
    LOG_FILE_NAME = "debug.log"
    tokenizer = ModBertTokenizer('base', cache_dir=config.cache_dir)
    label_token_dict = {
            "pad_token": "<pad>",
            "bos_token": "<t>",
            "eos_token": "</t>",}
    label_token_dict.update({
        f"label_{label_idx}_token": label 
        for label_idx, label in enumerate(config.joint_da_seg_recog_labels)
    })
    label_tokenizer = CustomizedTokenizer(
        token_dict=label_token_dict
    )

    # metrics calculator
    metrics = DAMetrics()
    model = SpeechTransformerLabeler(config, tokenizer, 
            label_tokenizer, freeze=config.freeze)
    model.load_model(config.model_path)
    print(f"model path: {config.model_path}")
    model.eval()

    for set_name in ["dev", "test"]:
        data_source = SpeechXTSource(
                split=set_name, config=config,
                tokenizer=tokenizer, label_tokenizer=label_tokenizer)
        mlog(str(data_source.statistics), config, LOG_FILE_NAME)
        current_score, metrics_results, split_loss = eval_split(
                model, data_source, set_name, 
                config, label_tokenizer, metrics, 
                LOG_FILE_NAME, write_pred=True)
        print("Split loss: ", split_loss)
        diff = (metrics_results['Macro F1'] - metrics_results['DER']) * 100

        lazy_s = f"DSER, DER, F1, LWER:\n {100*metrics_results['DSER']}\t{100*metrics_results['DER']}\t{100*metrics_results['Macro F1']}\t{diff}\t{100*metrics_results['Macro LWER']}\n"
        mlog(lazy_s, config, LOG_FILE_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_train", type=str2bool, default=True)
    parser.add_argument("--run_test", type=str2bool, default=False)

    # model - architecture
    parser.add_argument("--model", type=str, default="speech_xt")
    parser.add_argument("--freeze", type=str, default="all", 
            help="[all, top_layer, none]")

    # model - numbers
    parser.add_argument("--history_len", type=int, default=3, 
            help="number of history sentences")
    parser.add_argument("--attr_embedding_dim", type=int, default=32)
    parser.add_argument("--encoder_hidden_dim", type=int, default=128)
    parser.add_argument("--n_encoder_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--kvdim", type=int, default=64)
    parser.add_argument("--hist_out", type=int, default=128,
            help="history encoder output dim")
    parser.add_argument("--pos_encoder_hidden_dim", type=int, 
            default=128, help="position encoder hidden dim, if mode absolute")
    parser.add_argument("--pos_mode", type=str, default="absolute",
            help="position encoding mode: relative | absolute ")
    parser.add_argument("--pos_comb", type=str, default="cat",
            help="position encoding combination: add(itive) | (con)cat ")
    parser.add_argument("--pooling_mode_cls_token", 
            type=str2bool, default=True)
    parser.add_argument("--pooling_mode_mean_tokens", 
            type=str2bool, default=True)
    parser.add_argument("--pooling_mode_max_tokens", 
            type=str2bool, default=True)

    # speech encoder params
    parser.add_argument("--d_pause_embedding", type=int, default=2)
    parser.add_argument("--d_speech", type=int, default=128, 
            help="speech encoder output dim")
    parser.add_argument("--fixed_word_length", type=int, default=50)
    parser.add_argument("--num_conv", type=int, default=32)
    parser.add_argument("--conv_sizes", type=str, default="5,10,25,50",
            help="CNN filter widths")
    parser.add_argument("--downsample", type=str2bool, default=True)
    parser.add_argument("--feature_types", type=str, default=None)
    parser.add_argument("--seq_max_len", type=int, default=512, 
            help="max utterance length for truncation")

    # training
    parser.add_argument("--seed", type=int, default=42, 
            help="random initialization seed")
    parser.add_argument("--dropout", type=float, default=0.2, 
            help="dropout probability")
    parser.add_argument("--l2_penalty", type=float, default=0.0001, 
            help="l2 penalty")
    parser.add_argument("--optimizer", type=str, default="adam", 
            help="optimizer")
    parser.add_argument("--init_lr", type=float, default=0.0008, 
            help="init learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-7, 
            help="minimum learning rate for early stopping")
    parser.add_argument("--lr_decay_rate", type=float, default=0.5)
    parser.add_argument("--gradient_clip", type=float, default=5.0, 
            help="gradient clipping")
    parser.add_argument("--n_epochs", type=int, default=20, 
            help="number of epochs for training")
    parser.add_argument("--use_pretrained_word_embedding", type=str2bool, 
            default=True)
    parser.add_argument("--batch_size", type=int, default=64, 
            help="batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=64,
            help="batch size for evaluation")

    # management
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--model_path", help="path to model")
    parser.add_argument("--config_path", help="path to config")
    parser.add_argument("--enable_log", type=str2bool, default=True)
    parser.add_argument("--save_model", type=str2bool, default=True)
    parser.add_argument("--check_loss_after_n_step", type=int, default=100)
    parser.add_argument("--validate_after_n_step", type=int, default=1000)
    parser.add_argument("--filename_note", type=str, 
            help="take a note in saved files' names")
    config = parser.parse_args()
    
    from swda_utils.config import SpeechConfig as Config

    # load corpus config
    corpus_config = Config()
    corpus_config_dict = {}

    # merge parse args with corpus config
    # priority: parse args > corpus config
    for k, v in corpus_config.__dict__.items():
        if not k.startswith("__") and k not in config.__dict__:
            corpus_config_dict[k] = v
    config.__dict__.update(corpus_config_dict)
    if "conv_sizes" in config:
        convs = config.conv_sizes.split(',')
        convs = [int(x) for x in convs]
        config.conv_sizes = convs
    
    frame_feat_types = set(["pitch", "mfcc", "fbank", "fb3"])
    if config.feature_types is not None:
        config.feature_types = config.feature_types.split(',')
        print(config.feature_types)
        config.frame_features = frame_feat_types.intersection(set(config.feature_types))
    else:
        config.feature_types = []
        config.frame_features = []

    # set random seeds
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    if config.run_train:
        run_train(config)

    if config.run_test:
        run_test(config)

    exit(0)

