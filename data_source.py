import copy
import collections
import math
import random
import json
import code
import pandas as pd
import numpy as np
import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# NOTE:
# "utterances" here correspond to "turns" in the speech data
# "segments/fragments" here correspond to "sentence" in the speech data

class SpeechXTSource():
    def __init__(self, split, config, tokenizer, label_tokenizer):
        # Attributes
        # Attributes from config
        frame_feat_types = set(["pitch", "mfcc", "fbank", "fb3"])
        self.feature_types = config.feature_types
        self.frame_features = frame_feat_types.intersection(set(self.feature_types)) 
        self.split = split
        self.suffix = config.suffix
        self.dataset_path = config.dataset_path
        self.history_len = config.history_len
        self.feature_dir = config.feature_dir
        self.feature_types = config.feature_types
        self.downsample = config.downsample
        self.debug = config.debug
        self.fixed_word_length = config.fixed_word_length
        if "pause" in self.feature_types:
            self.pause_vocab = config.pause_vocab
        
        # Other attributes
        self.label_tokenizer = label_tokenizer
        self.tokenizer = tokenizer
        self.pad_label_id = label_tokenizer.pad_token_id
        self.statistics = {
                "n_sessions": 0, 
                "n_turns": 0, 
                "n_tokens": 0
                }
        self.fragments = self.get_samples()
        
        empty_sent = ""
        self.empty_tokens = self.tokenizer.convert_string_to_tokens(empty_sent)
        self.empty_ids = self.tokenizer.convert_tokens_to_ids(self.empty_tokens)
        self.empty_floor = 0
        self.empty_labels = [self.pad_label_id] * len(self.empty_ids)

    def __len__(self):
        return len(self.fragments)
    
    def get_dialog_keys(self):
        return self.dialog_keys

    def get_dialog_length(self, dialog_id):
        return len(self.fragments[dialog_id])

    def load_frames(self, dialog_id):
        data = {}
        speakers = ["A", "B"]
        
        for speaker in speakers:
            idx = speakers.index(speaker)
            data[idx] = []
            for feat in self.frame_features:
                filepath = os.path.join(self.feature_dir, feat + '_json')
                filename = os.path.join(filepath, 'sw' + dialog_id + \
                        '-' + speaker + '.json')
                with open(filename) as fIn:
                    frames = json.load(fIn)
                if self.downsample:
                    frames = frames[::2]
                if not data[idx]:
                    data[idx] = frames
                else:
                    # zip keeps everything and truncates at end
                    data[idx] = [x+y for x,y in zip(data[idx],frames)]
        self.feat_dim = len(data[0][0])
        return data

    def process_sent_frames(self, sent_partition, sent_frames):
        feat_dim = sent_frames.shape[0]
        speech_frames = []
        for frame_idx in sent_partition:
            center_frame = (frame_idx[0] + frame_idx[1])//2
            start_idx = center_frame - self.fixed_word_length//2
            end_idx = center_frame + self.fixed_word_length//2
            raw_word_frames = sent_frames[:, frame_idx[0]:frame_idx[1]]
            # feat_dim * number of frames
            raw_count = raw_word_frames.shape[1]
            if raw_count > self.fixed_word_length:
                # too many frames
                this_word_frames = sent_frames[:, frame_idx[0]:frame_idx[1]]
                extra_ratio = raw_count//self.fixed_word_length
                if extra_ratio < 2:  # delete things in the middle
                    mask = np.ones(raw_count, dtype=bool)
                    num_extra = raw_count - self.fixed_word_length
                    not_include = range(center_frame-num_extra,
                                        center_frame+num_extra)[::2]
                    # need to offset by beginning frame, since we are masking
                    # this_word_frames
                    not_include = [x-frame_idx[0] for x in not_include]
                    mask[not_include] = False
                else:  # too big, just sample
                    mask = np.zeros(raw_count, dtype=bool)
                    include = range(frame_idx[0], frame_idx[1])[::extra_ratio]
                    include = [x-frame_idx[0] for x in include]
                    if len(include) > self.fixed_word_length:
                        # still too many frames
                        num_current = len(include)
                        sub_extra = num_current - self.fixed_word_length
                        num_start = (num_current - sub_extra)//2
                        not_include = include[num_start:num_start+sub_extra]
                        for ni in not_include:
                            include.remove(ni)
                    mask[include] = True
                this_word_frames = this_word_frames[:, mask]
            else:  # not enough frames, choose frames extending from center
                this_word_frames = sent_frames[:, max(0, start_idx):end_idx]
                if this_word_frames.shape[1] == 0:
                    # make 0 if no frame info
                    this_word_frames = np.zeros((feat_dim, 
                        self.fixed_word_length))
                if start_idx < 0 and \
                        this_word_frames.shape[1] < self.fixed_word_length:
                    this_word_frames = np.hstack([
                        np.zeros((feat_dim, -start_idx)), this_word_frames])

                # still not enough frames
                if this_word_frames.shape[1] < self.fixed_word_length:
                    num_more = self.fixed_word_length-this_word_frames.shape[1]
                    this_word_frames = np.hstack(
                        [this_word_frames, np.zeros((feat_dim, num_more))])
            speech_frames.append(this_word_frames)
        
        return speech_frames

    def get_samples(self):
        """Creates examples sets."""
        filename = os.path.join(self.dataset_path, self.split + self.suffix)
        with open(filename, 'r') as f:
            sessions = json.load(f)

        # Just load 2 conversations for debugging
        if self.debug:
            temp = {}
            i = 0
            for filenumstr, sess in sessions.items():
                if i > 2: 
                    break
                temp[filenumstr] = sess
                i += 1
            del sessions
            sessions = temp

        self.dialog_keys = list(sessions.keys())
        
        processed_sessions = {}
        # Process sessions
        for filenumstr, sess in sessions.items():
            self.statistics["n_sessions"] += 1
            processed_sessions[filenumstr] = []

            for turn in sess:
                self.statistics["n_turns"] += 1
                tokens = turn["da_turn"]
                self.statistics["n_tokens"] += len(tokens)
                floor = turn["speaker"]
                turn_labels = turn["joint_labels"]
                bound_labels = []
                for x in turn_labels:
                    if x == "I":
                        bound_labels.append(0)
                    else:
                        bound_labels.append(1)
                turn_tokens = [self.tokenizer.cls_token] + \
                        tokens + [self.tokenizer.sep_token]
                turn_token_ids = self.tokenizer.convert_tokens_to_ids(\
                        turn_tokens)
                bound_label_ids = [bound_labels[0]] + bound_labels \
                        + [bound_labels[-1]]
                # For turns, CLS token gets "I" label, and SEP gets "PAD" label
                turn_label_ids = [self.label_tokenizer.bos_token_id]\
                        + [self.label_tokenizer.word2id[label] for \
                        label in turn_labels] \
                        + [self.label_tokenizer.eos_token_id] 
                turn_floor_id = ["A", "B"].index(floor)
                if "pause" in self.feature_types:
                    pause_before = [self.pause_vocab["<START>"]] + \
                        [self.pause_vocab[i] for i in turn['pause_before']] + \
                        [self.pause_vocab["<STOP>"]]
                    pause_after = [self.pause_vocab["<START>"]] + \
                        [self.pause_vocab[i] for i in turn['pause_after']] + \
                        [self.pause_vocab["<STOP>"]]
                else:
                    pause_before = []
                    pause_after = []
                # use first and last for START & STOP
                if self.frame_features:
                    partitions = [turn['partition'][0]] + \
                        turn['partition'] + [turn['partition'][-1]]
                    if self.downsample:
                        partitions = [(x//2, y//2) for x,y in partitions]
                else:
                    partitions = []
                if "pause_raw" in self.feature_types:
                    pauses_raw = np.array([turn['rp_before'], turn['rp_after']])
                    pauses_raw = np.hstack([np.zeros((2, 1)),\
                            pauses_raw, np.zeros((2, 1))])
                else:
                    pauses_raw = []
                if "word_dur" in self.feature_types:
                    word_durations = np.array([turn['dur_mean'], 
                        turn['dur_max']])
                    word_durations = np.hstack([np.zeros((2, 1)), 
                        word_durations, np.zeros((2, 1))])
                else:
                    word_durations = []

                processed_sessions[filenumstr].append({
                    "token_ids": turn_token_ids,
                    "label_ids": turn_label_ids,
                    "bound_ids": bound_label_ids,
                    "floor_id": turn_floor_id,
                    "pause_before_ids": pause_before,
                    "pause_after_ids": pause_after,
                    "partitions": partitions,
                    "word_durations": word_durations,
                    "pauses_raw": pauses_raw
                })  

        # Get segments
        fragments = {}
        for filenumstr, uttrs in processed_sessions.items():
            fragments[filenumstr] = []
            for uttr_end_idx in range(0, len(uttrs)):
                uttr_start_idx = max(0, uttr_end_idx-self.history_len+1)
                contexted_turn = uttrs[uttr_start_idx:uttr_end_idx+1]
                fragments[filenumstr].append(contexted_turn)
        return fragments

    # Note that get_batch_faetures here assumes a known dialog, thus frames are
    # already loaded
    def get_batch_features(self, dialog_key, all_dialog_frames, turn_indices):
        # batch features
        X, Y, X_floor, Y_B = [], [], [], []
        X_frames, X_pb, X_pf = [], [], []
        X_rp, X_wd = [], []

        for idx in turn_indices:
            segments = self.fragments[dialog_key][idx]
            # First non-padding segments
            this_X, this_X_floor = [], []
            this_frames, this_pb, this_pf = [], [], []
            this_rp, this_wd = [], []
            for segment in segments:
                this_X.append(segment["token_ids"])
                this_X_floor.append(segment["floor_id"])
                if self.frame_features:
                    side = np.array(all_dialog_frames[segment["floor_id"]]).T
                    seg_frames = self.process_sent_frames(segment["partitions"], side)
                    seg_frames = [word_frames.T for word_frames in seg_frames]
                    this_frames.append(seg_frames)
                if "pause" in self.feature_types:
                    this_pb.append(segment["pause_before_ids"]) 
                    this_pf.append(segment["pause_after_ids"])
                if "pause_raw" in self.feature_types:
                    this_rp.append(segment["pauses_raw"])
                if "word_dur" in self.feature_types:
                    this_wd.append(segment["word_durations"])
            segment = segments[-1]
            Y.append(segment["label_ids"])
            Y_B.append(segment["bound_ids"])
            # Then padding segments
            for _ in range(self.history_len-len(segments)):
                this_X.insert(0, self.empty_ids)
                this_X_floor.insert(0, self.empty_floor)
                this_frames.insert(0, [])
                this_pb.insert(0, self.empty_ids)
                this_pf.insert(0, self.empty_ids)
                this_rp.insert(0, np.zeros((2,1)))
                this_wd.insert(0, np.zeros((2,1)))

            X += this_X
            X_floor += this_X_floor
            X_frames += this_frames 
            X_pb += this_pb
            X_pf += this_pf
            X_rp += this_rp
            X_wd += this_wd

        X_ba = self.tokenizer.convert_batch_ids_to_tensor(X, self.history_len)
        X, X_type_ids, X_attn_masks = X_ba
        max_segment_len = X.size(1)
        batch_size = len(turn_indices)
        if "pause" in self.feature_types:
            padded_pb = [ids + 
                [self.tokenizer.pad_token_id]*(max_segment_len-len(ids)) 
                for ids in X_pb]
            padded_pf = [ids + 
                [self.tokenizer.pad_token_id]*(max_segment_len-len(ids)) 
                for ids in X_pf]
            pb_tensor = torch.LongTensor(padded_pb).to(DEVICE)
            pf_tensor = torch.LongTensor(padded_pf).to(DEVICE)
            pause_feats = [pb_tensor, pf_tensor]
        else:
            pause_feats = None

        scalar_feats = [None, None]
        if "pause_raw" in self.feature_types:
            padded_rp = [np.hstack([feats, 
                np.zeros((2, max_segment_len-feats.shape[1]))]) 
                for feats in X_rp]
            scalar_feats[0] = torch.Tensor(padded_rp).to(DEVICE)

        if "word_dur" in self.feature_types:
            padded_wd = [np.hstack([feats, 
                np.zeros((2, max_segment_len-feats.shape[1]))]) 
                for feats in X_wd]
            scalar_feats[1] = torch.Tensor(padded_wd).to(DEVICE)

        if self.frame_features:
            padded_frames = [feats + 
                [np.zeros((self.fixed_word_length, self.feat_dim))] * (
                max_segment_len-len(feats)) for feats in X_frames]
            frame_feats = padded_frames
        else:
            frame_feats = None

        Y = [y + [self.pad_label_id]*(max_segment_len-len(y)) for y in Y]
        Y_B = [y + [self.pad_label_id]*(max_segment_len-len(y)) for y in Y_B]
        X_floor = torch.LongTensor(X_floor).view(batch_size, 
                self.history_len)
        Y = torch.LongTensor(Y).view(batch_size, -1)
        Y_B = torch.LongTensor(Y_B).view(batch_size, -1)

        X_type_ids = X_type_ids
        X_attn_masks = X_attn_masks
        batch_data_dict = {
            "X": X.to(DEVICE),
            "X_floor": X_floor.to(DEVICE),
            "X_type_ids": X_type_ids.to(DEVICE),
            "X_attn_masks": X_attn_masks.to(DEVICE),
            "X_speech": [pause_feats, frame_feats, scalar_feats],
            "Y": Y.to(DEVICE),
            "Y_B": Y_B.to(DEVICE)
        }

        return batch_data_dict

# Ideally tokenizer should be inferred/consistent with features 
# TODO for later cleanup
class SpeechDataSource():
    def __init__(self, split, config, tokenizer, label_tokenizer):
        # Attributes
        # Attributes from config
        frame_feat_types = set(["pitch", "mfcc", "fbank", "fb3"])
        self.feature_types = config.feature_types
        self.frame_features = frame_feat_types.intersection(set(self.feature_types)) 
        self.split = split
        self.suffix = config.suffix
        self.dataset_path = config.dataset_path
        self.history_len = config.history_len
        self.feature_dir = config.feature_dir
        self.feature_types = config.feature_types
        self.downsample = config.downsample
        self.fixed_word_length = config.fixed_word_length
        if "pause" in self.feature_types:
            self.pause_vocab = config.pause_vocab
        
        # Other attributes
        self.label_tokenizer = label_tokenizer
        self.tokenizer = tokenizer
        self.pad_label_id = label_tokenizer.pad_token_id
        self.bos_label_id = label_tokenizer.bos_token_id
        self.eos_label_id = label_tokenizer.eos_token_id
        self.statistics = {
                "n_sessions": 0, 
                "n_turns": 0, 
                "n_tokens": 0
                }
        self.fragments = self.get_samples()
        
    def __len__(self):
        return len(self.fragments)
    
    def get_dialog_keys(self):
        return self.dialog_keys

    def get_dialog_length(self, dialog_id):
        return len(self.fragments[dialog_id])

    def load_frames(self, dialog_id):
        data = {}
        speakers = ["A", "B"]
        
        for speaker in speakers:
            idx = speakers.index(speaker)
            data[idx] = []
            for feat in self.frame_features:
                filepath = os.path.join(self.feature_dir, feat + '_json')
                filename = os.path.join(filepath, 'sw' + dialog_id + \
                        '-' + speaker + '.json')
                with open(filename) as fIn:
                    frames = json.load(fIn)
                if self.downsample:
                    frames = frames[::2]
                if not data[idx]:
                    data[idx] = frames
                else:
                    # zip keeps everything and truncates at end
                    data[idx] = [x+y for x,y in zip(data[idx],frames)]
        self.feat_dim = len(data[0][0])
        return data

    def process_sent_frames(self, sent_partition, sent_frames):
        feat_dim = sent_frames.shape[0]
        speech_frames = []
        for frame_idx in sent_partition:
            center_frame = (frame_idx[0] + frame_idx[1])//2
            start_idx = center_frame - self.fixed_word_length//2
            end_idx = center_frame + self.fixed_word_length//2
            raw_word_frames = sent_frames[:, frame_idx[0]:frame_idx[1]]
            # feat_dim * number of frames
            raw_count = raw_word_frames.shape[1]
            if raw_count > self.fixed_word_length:
                # too many frames
                this_word_frames = sent_frames[:, frame_idx[0]:frame_idx[1]]
                extra_ratio = raw_count//self.fixed_word_length
                if extra_ratio < 2:  # delete things in the middle
                    mask = np.ones(raw_count, dtype=bool)
                    num_extra = raw_count - self.fixed_word_length
                    not_include = range(center_frame-num_extra,
                                        center_frame+num_extra)[::2]
                    # need to offset by beginning frame, since we are masking
                    # this_word_frames
                    not_include = [x-frame_idx[0] for x in not_include]
                    mask[not_include] = False
                else:  # too big, just sample
                    mask = np.zeros(raw_count, dtype=bool)
                    include = range(frame_idx[0], frame_idx[1])[::extra_ratio]
                    include = [x-frame_idx[0] for x in include]
                    if len(include) > self.fixed_word_length:
                        # still too many frames
                        num_current = len(include)
                        sub_extra = num_current - self.fixed_word_length
                        num_start = (num_current - sub_extra)//2
                        not_include = include[num_start:num_start+sub_extra]
                        for ni in not_include:
                            include.remove(ni)
                    mask[include] = True
                this_word_frames = this_word_frames[:, mask]
            else:  # not enough frames, choose frames extending from center
                this_word_frames = sent_frames[:, max(0, start_idx):end_idx]
                if this_word_frames.shape[1] == 0:
                    # make 0 if no frame info
                    this_word_frames = np.zeros((feat_dim, 
                        self.fixed_word_length))
                if start_idx < 0 and \
                        this_word_frames.shape[1] < self.fixed_word_length:
                    this_word_frames = np.hstack([
                        np.zeros((feat_dim, -start_idx)), this_word_frames])

                # still not enough frames
                if this_word_frames.shape[1] < self.fixed_word_length:
                    num_more = self.fixed_word_length-this_word_frames.shape[1]
                    this_word_frames = np.hstack(
                        [this_word_frames, np.zeros((feat_dim, num_more))])
            speech_frames.append(this_word_frames)
        
        return speech_frames

    def get_samples(self):
        """Creates examples sets."""
        filename = os.path.join(self.dataset_path, self.split + self.suffix)
        with open(filename, 'r') as f:
            sessions = json.load(f)

        self.dialog_keys = list(sessions.keys())
        
        processed_sessions = {}
        # Process sessions
        for filenumstr, sess in sessions.items():
            self.statistics["n_sessions"] += 1
            processed_sessions[filenumstr] = []

            for turn in sess:
                self.statistics["n_turns"] += 1
                tokens = turn["da_turn"]
                self.statistics["n_tokens"] += len(tokens)
                floor = turn["speaker"]
                turn_labels = turn["joint_labels"]
                bound_labels = []
                for x in turn_labels:
                    if x == "I":
                        bound_labels.append(0)
                    else:
                        bound_labels.append(1)
                turn_tokens = [self.tokenizer.cls_token] + \
                        tokens + [self.tokenizer.sep_token]
                turn_token_ids = self.tokenizer.convert_tokens_to_ids(\
                        turn_tokens)
                bound_label_ids = [self.bos_label_id] + bound_labels + [self.eos_label_id]
                turn_label_ids = [self.bos_label_id] + [self.label_tokenizer.word2id[label] for label in turn_labels] + [self.eos_label_id]
                turn_floor_id = ["A", "B"].index(floor)
                if "pause" in self.feature_types:
                    pause_before = [self.pause_vocab["<START>"]] + \
                        [self.pause_vocab[i] for i in turn['pause_before']] + \
                        [self.pause_vocab["<STOP>"]]
                    pause_after = [self.pause_vocab["<START>"]] + \
                        [self.pause_vocab[i] for i in turn['pause_after']] + \
                        [self.pause_vocab["<STOP>"]]
                else:
                    pause_before = []
                    pause_after = []
                # use first and last for START & STOP
                if self.frame_features:
                    partitions = [turn['partition'][0]] + \
                        turn['partition'] + [turn['partition'][-1]]
                    if self.downsample:
                        partitions = [(x//2, y//2) for x,y in partitions]
                else:
                    partitions = []
                if "pause_raw" in self.feature_types:
                    pauses_raw = np.array([turn['rp_before'], turn['rp_after']])
                    pauses_raw = np.hstack([np.zeros((2, 1)),\
                            pauses_raw, np.zeros((2, 1))])
                else:
                    pauses_raw = []
                if "word_dur" in self.feature_types:
                    word_durations = np.array([turn['dur_mean'], 
                        turn['dur_max']])
                    word_durations = np.hstack([np.zeros((2, 1)), 
                        word_durations, np.zeros((2, 1))])
                else:
                    word_durations = []

                processed_sessions[filenumstr].append({
                    "token_ids": turn_token_ids,
                    "label_ids": turn_label_ids,
                    "bound_ids": bound_label_ids,
                    "floor_id": turn_floor_id,
                    "pause_before_ids": pause_before,
                    "pause_after_ids": pause_after,
                    "partitions": partitions,
                    "word_durations": word_durations,
                    "pauses_raw": pauses_raw
                })  

        # Get segments
        fragments = {}
        for filenumstr, uttrs in processed_sessions.items():
            fragments[filenumstr] = []
            for uttr_end_idx in range(0, len(uttrs)):
                uttr_start_idx = max(0, uttr_end_idx-self.history_len+1)
                contexted_turn = uttrs[uttr_start_idx:uttr_end_idx+1]
                fragments[filenumstr].append(contexted_turn)
        return fragments

    # Note that get_batch_faetures here assumes a known dialog, thus frames are
    # already loaded
    def get_batch_features(self, dialog_key, all_dialog_frames, turn_indices):
        # batch features
        X, Y, X_floor, Y_B = [], [], [], []
        X_frames, X_pb, X_pf = [], [], []
        X_rp, X_wd = [], []

        empty_sent = ""
        empty_tokens = self.tokenizer.convert_string_to_tokens(empty_sent)
        empty_ids = self.tokenizer.convert_tokens_to_ids(empty_tokens)
        empty_floor = 0
        empty_labels = [self.pad_label_id] * len(empty_ids)

        for idx in turn_indices:
            segments = self.fragments[dialog_key][idx]
            # First non-padding segments
            this_X, this_X_floor = [], []
            this_frames, this_pb, this_pf = [], [], []
            this_rp, this_wd = [], []
            for segment in segments:
                this_X.append(segment["token_ids"])
                this_X_floor.append(segment["floor_id"])
                if self.frame_features:
                    side = np.array(all_dialog_frames[segment["floor_id"]]).T
                    seg_frames = self.process_sent_frames(segment["partitions"], side)
                    seg_frames = [word_frames.T for word_frames in seg_frames]
                    this_frames.append(seg_frames)
                if "pause" in self.feature_types:
                    this_pb.append(segment["pause_before_ids"]) 
                    this_pf.append(segment["pause_after_ids"])
                if "pause_raw" in self.feature_types:
                    this_rp.append(segment["pauses_raw"])
                if "word_dur" in self.feature_types:
                    this_wd.append(segment["word_durations"])
            segment = segments[-1]
            Y.append(segment["label_ids"])
            Y_B.append(segment["bound_ids"])
            # Then padding segments
            for _ in range(self.history_len-len(segments)):
                this_X.insert(0, empty_ids)
                this_X_floor.insert(0, empty_floor)
                this_frames.insert(0, [])
                this_pb.insert(0, empty_ids)
                this_pf.insert(0, empty_ids)
                this_rp.insert(0, np.zeros((2,1)))
                this_wd.insert(0, np.zeros((2,1)))

            X += this_X
            X_floor += this_X_floor
            X_frames += this_frames 
            X_pb += this_pb
            X_pf += this_pf
            X_rp += this_rp
            X_wd += this_wd

        X_ba = self.tokenizer.convert_batch_ids_to_tensor(X, self.history_len)
        X, X_type_ids, X_attn_masks = X_ba
        max_segment_len = X.size(1)
        batch_size = len(turn_indices)
        if "pause" in self.feature_types:
            padded_pb = [ids + 
                [self.tokenizer.pad_token_id]*(max_segment_len-len(ids)) 
                for ids in X_pb]
            padded_pf = [ids + 
                [self.tokenizer.pad_token_id]*(max_segment_len-len(ids)) 
                for ids in X_pf]
            pb_tensor = torch.LongTensor(padded_pb).to(DEVICE)
            pf_tensor = torch.LongTensor(padded_pf).to(DEVICE)
            pause_feats = [pb_tensor, pf_tensor]
        else:
            pause_feats = None

        scalar_feats = [None, None]
        if "pause_raw" in self.feature_types:
            padded_rp = [np.hstack([feats, 
                np.zeros((2, max_segment_len-feats.shape[1]))]) 
                for feats in X_rp]
            scalar_feats[0] = torch.Tensor(padded_rp).to(DEVICE)

        if "word_dur" in self.feature_types:
            padded_wd = [np.hstack([feats, 
                np.zeros((2, max_segment_len-feats.shape[1]))]) 
                for feats in X_wd]
            scalar_feats[1] = torch.Tensor(padded_wd).to(DEVICE)

        if self.frame_features:
            padded_frames = [feats + 
                [np.zeros((self.fixed_word_length, self.feat_dim))] * (
                max_segment_len-len(feats)) for feats in X_frames]
            frame_feats = padded_frames
        else:
            frame_feats = None

        Y = [y + [self.pad_label_id]*(max_segment_len-len(y)) for y in Y]
        Y_B = [y + [self.pad_label_id]*(max_segment_len-len(y)) for y in Y_B]
        X_floor = torch.LongTensor(X_floor).view(batch_size, 
                self.history_len)
        Y = torch.LongTensor(Y).view(batch_size, -1)
        Y_B = torch.LongTensor(Y_B).view(batch_size, -1)

        X_type_ids = X_type_ids
        X_attn_masks = X_attn_masks
        batch_data_dict = {
            "X": X.to(DEVICE),
            "X_floor": X_floor.to(DEVICE),
            "X_type_ids": X_type_ids.to(DEVICE),
            "X_attn_masks": X_attn_masks.to(DEVICE),
            "X_speech": [pause_feats, frame_feats, scalar_feats],
            "Y": Y.to(DEVICE),
            "Y_B": Y_B.to(DEVICE)
        }

        return batch_data_dict


class TextDataSource():
    def __init__(self, data, config, tokenizer, label_tokenizer):
        # Attributes
        # Attributes from config
        self.dataset_path = config.dataset_path
        self.max_uttr_len = config.max_uttr_len
        self.history_len = config.history_len
        # Other attributes
        self.label_tokenizer = label_tokenizer
        self.tokenizer = tokenizer
        self.pad_label_id = label_tokenizer.pad_token_id
        self.bos_label_id = label_tokenizer.bos_token_id
        self.eos_label_id = label_tokenizer.eos_token_id
        self.statistics = {"n_sessions": 0, "n_uttrs": 0, "n_tokens": 0, "n_segments": 0, "n_fragments": 0}

        sessions = data

        # Process sessions
        for sess in sessions:
            sess["processed_utterances"] = []
            for uttr in sess["utterances"]:
                uttr_tokens = []
                uttr_labels = []
                bound_label_ids = []
                for segment in uttr:
                    text = segment["text"]
                    floor = segment["floor"]
                    dialog_act = segment["segment_meta"]["dialog_act"]
                    tokens = self.tokenizer.convert_string_to_tokens(text)[:self.max_uttr_len]
                    uttr_tokens += tokens
                    uttr_labels += ["I"] * (len(tokens) - 1) + ["E_"+dialog_act]
                    bound_label_ids += [0] * (len(tokens) - 1) + [1]

                uttr_tokens = [self.tokenizer.cls_token] + uttr_tokens + [self.tokenizer.sep_token]
                uttr_token_ids = self.tokenizer.convert_tokens_to_ids(uttr_tokens)
                uttr_label_ids = [self.bos_label_id] + [self.label_tokenizer.word2id[label] for label in uttr_labels] + [self.eos_label_id]
                bound_label_ids = [self.bos_label_id] + bound_label_ids + [self.eos_label_id]
                uttr_floor_id = ["A", "B"].index(floor)
                
                sess["processed_utterances"].append({
                    "token_ids": uttr_token_ids,
                    "label_ids": uttr_label_ids,
                    "floor_id": uttr_floor_id,
                    "bound_ids": bound_label_ids,
                })  

        # Get segments
        self._fragments = []
        for sess in sessions:
            uttrs = sess["processed_utterances"]
            for uttr_end_idx in range(0, len(uttrs)):
                uttr_start_idx = max(0, uttr_end_idx-self.history_len+1)
                fragment = {
                    "utterances": uttrs[uttr_start_idx:uttr_end_idx+1],
                }
                self._fragments.append(fragment)

        # Calculate basic statistics
        self.statistics["n_sessions"] = len(sessions)
        self.statistics["n_fragments"] = len(self._fragments)
        for sess in sessions:
            self.statistics["n_uttrs"] += len(sess["utterances"])
            for uttr in sess["utterances"]:
                self.statistics["n_segments"] += len(uttr)
                for segment in uttr:
                    tokens = segment["text"].split(" ")
                    self.statistics["n_tokens"] += len(tokens)

    def epoch_init(self, shuffle=False):
        self.cur_fragment_idx = 0
        if shuffle:
            self.fragments = copy.deepcopy(self._fragments)
            random.shuffle(self.fragments)
        else:
            self.fragments = self._fragments

    def __len__(self):
        return len(self._fragments)

    # NOTE: for bert-based tokenizers, add pad/cls/attention ids 
    # Similar to sentence_transformers/src/models/BERT.py --> get_sentence_features
    def next(self, batch_size):
        # Return None when running out of segments
        if self.cur_fragment_idx == len(self.fragments):
            return None

        # Data to fill in
        X, X_floor = [], []
        Y, Y_B = [], []

        empty_sent = ""
        empty_tokens = self.tokenizer.convert_string_to_tokens(empty_sent)
        empty_ids = self.tokenizer.convert_tokens_to_ids(empty_tokens)
        padding_segment = {
            "tokens": empty_tokens,
            "token_ids": empty_ids,
            "floor_id": 0,
            "label_ids": [self.pad_label_id] * len(empty_ids),
            "bound_ids": [self.pad_label_id] * len(empty_ids)
        }

        while self.cur_fragment_idx < len(self.fragments):
            if len(Y) == batch_size:
                break

            fragment = self.fragments[self.cur_fragment_idx]
            segments = fragment["utterances"]
            self.cur_fragment_idx += 1

            # First non-padding segments
            this_X, this_X_floor = [], []
            for segment in segments:
                this_X.append(segment["token_ids"])
                this_X_floor.append(segment["floor_id"])
            
            segment = segments[-1]
            Y.append(segment["label_ids"])
            Y_B.append(segment["bound_ids"])

            # Then padding segments
            for _ in range(self.history_len-len(segments)):
            #for _ in range(history_len-len(segments)):
                segment = padding_segment
                this_X.insert(0, segment["token_ids"])
                this_X_floor.insert(0, segment["floor_id"])

            X += this_X
            X_floor += this_X_floor

        X, X_type_ids, X_attn_masks = self.tokenizer.convert_batch_ids_to_tensor(X, self.history_len)
        max_segment_len = X.size(1)
        # This is needed because of batches that might be smaller 
        # history_len should not be affected
        batch_size = len(Y)
        Y = [y + [self.pad_label_id]*(max_segment_len-len(y)) for y in Y]
        Y_B = [y + [self.pad_label_id]*(max_segment_len-len(y)) for y in Y_B]
        X_floor = torch.LongTensor(X_floor).to(DEVICE).view(batch_size, self.history_len)
        Y = torch.LongTensor(Y).to(DEVICE).view(batch_size, -1)
        Y_B = torch.LongTensor(Y_B).to(DEVICE).view(batch_size, -1)

        # NOTE: don't do this reshape here
        #X = X.to(DEVICE).view(batch_size, self.history_len, -1)
        X = X.to(DEVICE) 
        X_type_ids = X_type_ids.to(DEVICE)
        X_attn_masks = X_attn_masks.to(DEVICE)
        batch_data_dict = {
            "X": X,
            "X_floor": X_floor,
            "X_type_ids": X_type_ids,
            "X_attn_masks": X_attn_masks,
            "Y": Y,
            "Y_B": Y_B
        }

        return batch_data_dict


class DataSource():

    def __init__(self, data, config, tokenizer, label_tokenizer):
        # Attributes
        # Attributes from config
        self.dataset_path = config.dataset_path
        self.max_uttr_len = config.max_uttr_len
        self.history_len = config.history_len
        # Other attributes
        self.label_tokenizer = label_tokenizer
        self.tokenizer = tokenizer
        self.pad_label_id = label_tokenizer.pad_token_id
        self.bos_label_id = label_tokenizer.bos_token_id
        self.eos_label_id = label_tokenizer.eos_token_id
        self.statistics = {"n_sessions": 0, "n_uttrs": 0, "n_tokens": 0, "n_segments": 0, "n_fragments": 0}

        sessions = data

        # Process sessions
        for sess in sessions:
            sess["processed_utterances"] = []
            for uttr in sess["utterances"]:
                uttr_tokens = []
                uttr_labels = []
                for segment in uttr:
                    text = segment["text"]
                    floor = segment["floor"]
                    dialog_act = segment["segment_meta"]["dialog_act"]
                    tokens = self.tokenizer.convert_string_to_tokens(text)[:self.max_uttr_len]
                    uttr_tokens += tokens
                    uttr_labels += ["I"] * (len(tokens) - 1) + ["E_"+dialog_act]

                uttr_token_ids = self.tokenizer.convert_tokens_to_ids(uttr_tokens, bos_and_eos=True)
                uttr_label_ids = [self.bos_label_id] + \
                    [self.label_tokenizer.word2id[label] for label in uttr_labels] + \
                    [self.eos_label_id]
                uttr_floor_id = ["A", "B"].index(floor)
                
                sess["processed_utterances"].append({
                    "token_ids": uttr_token_ids,
                    "label_ids": uttr_label_ids,
                    "floor_id": uttr_floor_id
                })  

        # Get segments
        self._fragments = []
        for sess in sessions:
            uttrs = sess["processed_utterances"]
            for uttr_end_idx in range(0, len(uttrs)):
                uttr_start_idx = max(0, uttr_end_idx-self.history_len+1)
                fragment = {
                    "utterances": uttrs[uttr_start_idx:uttr_end_idx+1],
                }
                self._fragments.append(fragment)

        # Calculate basic statistics
        self.statistics["n_sessions"] = len(sessions)
        self.statistics["n_fragments"] = len(self._fragments)
        for sess in sessions:
            self.statistics["n_uttrs"] += len(sess["utterances"])
            for uttr in sess["utterances"]:
                self.statistics["n_segments"] += len(uttr)
                for segment in uttr:
                    tokens = segment["text"].split(" ")
                    self.statistics["n_tokens"] += len(tokens)

    def epoch_init(self, shuffle=False):
        self.cur_fragment_idx = 0
        if shuffle:
            self.fragments = copy.deepcopy(self._fragments)
            random.shuffle(self.fragments)
        else:
            self.fragments = self._fragments

    def __len__(self):
        return len(self._fragments)

    def next(self, batch_size):
        # Return None when running out of segments
        if self.cur_fragment_idx == len(self.fragments):
            return None

        # Data to fill in
        X, Y = [], []
        X_floor = []

        empty_sent = ""
        empty_tokens = self.tokenizer.convert_string_to_tokens(empty_sent)
        empty_ids = self.tokenizer.convert_tokens_to_ids(empty_tokens, bos_and_eos=False)
        padding_segment = {
            "tokens": empty_tokens,
            "token_ids": empty_ids,
            "floor_id": 0,
            "label_ids": [self.pad_label_id] * len(empty_ids)
        }

        while self.cur_fragment_idx < len(self.fragments):
            if len(Y) == batch_size:
                break

            fragment = self.fragments[self.cur_fragment_idx]
            segments = fragment["utterances"]
            self.cur_fragment_idx += 1

            # First non-padding segments
            for segment in segments:
                X.append(segment["token_ids"])
                X_floor.append(segment["floor_id"])
            
            segment = segments[-1]
            Y.append(segment["label_ids"])

            # Then padding segments
            for _ in range(self.history_len-len(segments)):
                segment = padding_segment
                X.append(segment["token_ids"])
                X_floor.append(segment["floor_id"])

        X = self.tokenizer.convert_batch_ids_to_tensor(X)
        max_segment_len = X.size(1)
        Y = [y + [self.pad_label_id]*(max_segment_len-len(y)) for y in Y]

        batch_size = len(Y)
        history_len = X.size(0)//batch_size

        X = torch.LongTensor(X).to(DEVICE).view(batch_size, history_len, -1)
        X_floor = torch.LongTensor(X_floor).to(DEVICE).view(batch_size, history_len)
        Y = torch.LongTensor(Y).to(DEVICE).view(batch_size, -1)

        batch_data_dict = {
            "X": X,
            "X_floor": X_floor,
            "Y": Y
        }

        return batch_data_dict
