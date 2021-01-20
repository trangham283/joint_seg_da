#!/usr/bin/env python

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

from utils.metrics import JointDAMetrics as DAMetrics
from tokenization.customized_tokenizer import CustomizedTokenizer
from tokenization.bert_tokenizer import ModBertTokenizer
from model.joint_da_seg_recog.ctx_attn_ed import SpeechAttnEDSeqLabeler, SpeechBaselineLabeler
from data_source import SpeechDataSource, SpeechXTSource

model_dir = "/g/ssli/data/CTS-English/swbd_align/joint_da_seg/models/"

def str2bool(v):
    return v.lower() in ('true', '1', "True")

def reslog(s, RES_FILE_NAME):
    if not os.path.exists(f"{config.task_data_dir}/asr_out/"):
        os.makedirs(f"{config.task_data_dir}/asr_out/")

    with open(f"{config.task_data_dir}/asr_out/debug_{RES_FILE_NAME}.res", "a+", 
            encoding="utf-8") as log_f:
        log_f.write(s+"\n")

def eval_split(model, data_source, set_name, config, tokenizer, label_tokenizer):
    RES_FILE_NAME = set_name + "_" + config.model_name
    s = "TURN_ID\tPREDS"
    reslog(s, RES_FILE_NAME)

    pred_labels = []
    for dialog_idx in data_source.dialog_keys:
        if config.frame_features:
            dialog_frames = data_source.load_frames(dialog_idx)
        else:
            dialog_frames = []
        dialog_length = data_source.get_dialog_length(dialog_idx)
        turn_keys = list(range(dialog_length))
        for offset in range(0, dialog_length, config.batch_size):
            turn_idx = turn_keys[offset:offset+config.batch_size]
            batch_data = data_source.get_batch_features(dialog_idx, dialog_frames, turn_idx)
            batch_size = batch_data['X'].size(0) // config.history_len
            X_data = batch_data['X'].view(batch_size, config.history_len, -1)
            #print(X_data.shape)
            lens = []
            for bidx in range(batch_size):
                l = (X_data[bidx, config.history_len-1, 1:] != tokenizer.pad_token_id).sum(-1)
                lens.append(l.item())
            sent_ids = batch_data["sent_ids"]
            #print(lens)
            
            # Forward
            ret_data, ret_stat = model.test_step(batch_data)
            if config.model in ["bert_attn_ed", "speech_attn_ed"]:
                hyps = ret_data["symbols"].tolist()
            else:
                hyps = ret_data["symbols"].squeeze(-1).tolist()
            for i, pred_label_ids in enumerate(hyps):
                end_idx = lens[i]
                pred_syms = [label_tokenizer.id2word[label_id] for label_id in pred_label_ids[:end_idx]]
                sent_id = sent_ids[i]
                assert len(sent_id) == 1
                turn_id = sent_id[0][:-2]
                #print(i, pred_syms)
                s = turn_id + "\t" + " ".join(pred_syms) 
                reslog(s, RES_FILE_NAME)
                pred_labels.append(pred_syms)
    outname = f"{config.task_data_dir}/asr_out/{RES_FILE_NAME}.res" 
    return outname

# TODO? Need to change tokenization of ASR data if so
def run_pred_csl(config):
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

    dev_data_source = DataSource(
        data=dataset["dev"],
        config=config,
        tokenizer=tokenizer,
        label_tokenizer=label_tokenizer
    )



def run_pred(config):
    # tokenizers
    tokenizer = ModBertTokenizer('base', cache_dir=config.cache_dir)

    if config.model in ["bert_attn_ed", "speech_attn_ed"]:
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
        print("----- Loading dev data -----")
        dev_data_source = SpeechDataSource(
            split="dev", 
            config=config,
            tokenizer=tokenizer,
            label_tokenizer=label_tokenizer
        )
        print(str(dev_data_source.statistics))

        print("----- Loading test data -----")
        test_data_source = SpeechDataSource(
            split="test", 
            config=config,
            tokenizer=tokenizer,
            label_tokenizer=label_tokenizer
        )
        print(str(test_data_source.statistics))
    elif config.model in ["speech_xt", "speech_bl"]:
        label_token_dict = {
            "pad_token": "<pad>",
            "bos_token": "<t>",
            "eos_token": "</t>"
        }
        label_token_dict.update({
            f"label_{label_idx}_token": label 
            for label_idx, label in enumerate(config.joint_da_seg_recog_labels)
        })
        label_tokenizer = CustomizedTokenizer(
            token_dict=label_token_dict
        )
        # data loaders & number reporters
        print("----- Loading dev data -----")
        dev_data_source = SpeechXTSource(
            split="dev", 
            config=config,
            tokenizer=tokenizer,
            label_tokenizer=label_tokenizer
        )
        print(str(dev_data_source.statistics))

        print("----- Loading test data -----")
        test_data_source = SpeechXTSource(
            split="test", 
            config=config,
            tokenizer=tokenizer,
            label_tokenizer=label_tokenizer
        )
        print(str(test_data_source.statistics))
    else:
        print("Invalid model")
        exit(0)


    # metrics calculator
    metrics = DAMetrics()

    
    # set up  model
    if config.model == "bert_attn_ed":
        Model = BertAttnEDSeqLabeler
    elif config.model == "speech_attn_ed":
        Model = SpeechAttnEDSeqLabeler
    elif config.model == "speech_xt":
        Model = SpeechTransformerLabeler
    elif config.model == "speech_bl":
        Model = SpeechBaselineLabeler
    else:
        print("no model specified")
        exit(0)
    model = Model(config, tokenizer, label_tokenizer, freeze=config.freeze)

    # model adaption
    if torch.cuda.is_available():
        print("----- Using GPU -----")
        model = model.cuda()

    model_name = config.model_name + ".model.pt"
    model_path = os.path.join(model_dir, model_name)
    model.load_model(model_path)
    print(f"model path: {model_path}")

    for set_name, data_source in [("DEV", dev_data_source), ("TEST", test_data_source)]:
        output = eval_split(model, data_source, set_name, config, tokenizer, label_tokenizer)
        print(f"Written to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model - architecture
    parser.add_argument("--model", type=str, default="speech_attn_ed")
    parser.add_argument("--loss_type", type=str, default="cross_entropy")
    parser.add_argument("--rnn_type", type=str, default="gru", 
            help="[gru, lstm]")
    parser.add_argument("--freeze", type=str, default="all", 
            help="[all, pooler_only, top_layer]")
    parser.add_argument("--tie_weights", type=str2bool, default=True, 
            help="tie weights for decoder")
    parser.add_argument("--attention_type", type=str, default="sent", 
            help="[word, sent]")

    # model - numbers
    parser.add_argument("--vocab_size", type=int, default=20000, 
            help="keep top frequent words; relevant to GloVe-like emb only")
    parser.add_argument("--history_len", type=int, default=3, 
            help="number of history sentences")
    parser.add_argument("--attr_embedding_dim", type=int, default=30)
    parser.add_argument("--sent_encoder_hidden_dim", type=int, default=100)
    parser.add_argument("--n_sent_encoder_layers", type=int, default=1)
    parser.add_argument("--dial_encoder_hidden_dim", type=int, default=200)
    parser.add_argument("--n_dial_encoder_layers", type=int, default=1)
    parser.add_argument("--decoder_hidden_dim", type=int, default=200)
    parser.add_argument("--n_decoder_layers", type=int, default=1)

    # speech encoder params
    parser.add_argument("--d_pause_embedding", type=int, default=2)
    parser.add_argument("--d_speech", type=int, default=128, 
            help="speech encoder output dim")
    parser.add_argument("--fixed_word_length", type=int, default=100)
    parser.add_argument("--num_conv", type=int, default=32)
    parser.add_argument("--conv_sizes", type=str, default="5,10,25,50",
            help="CNN filter widths")
    parser.add_argument("--downsample", type=str2bool, default=False)
    parser.add_argument("--feature_types", type=str, default=None)
    parser.add_argument("--seq_max_len", type=int, default=512, 
            help="max utterance length for truncation")
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
            
    # other
    parser.add_argument("--seed", type=int, default=42, 
            help="random initialization seed")
    parser.add_argument("--max_uttr_len", type=int, default=45, 
            help="max utterance length for truncation")
    parser.add_argument("--batch_size", type=int, default=64, 
            help="batch size for training")

    # inference
    parser.add_argument("--decode_max_len", type=int, default=100, 
            help="max utterance length for decoding")
    parser.add_argument("--gen_type", type=str, default="greedy", 
            help="[greedy, sample, top]")
    parser.add_argument("--temp", type=float, default=1.0, 
            help="temperature for decoding")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.0)

    # management
    parser.add_argument("--debug", default=False)
    parser.add_argument("--model_name", help="name of model", required=True)
    parser.add_argument("--filename_note", type=str, 
            help="take a note in saved files' names")
    config = parser.parse_args()

    # load corpus config
    from swda_utils.config import ASRConfig 
    corpus_config = ASRConfig()

    # merge parse args with corpus config
    # priority: parse args > corpus config
    corpus_config_dict = {}
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

    run_pred(config)

    exit(0)

