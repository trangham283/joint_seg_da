import copy
import collections
import math
import random
import json
import code
import pandas as pd
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO:
# "utterances" here correspond to "turns" in the speech data
# "segments/fragments" here correspond to "sentence" in the speech data

# Ideally tokenizer should be inferred/consistent with features 
# TODO for later cleanup
class SpeechDataSource():
    def __init__(self, data, config, tokenizer, label_tokenizer):
        # Attributes
        # Attributes from config
        self.dataset_path = config.dataset_path
        self.max_uttr_len = config.max_uttr_len
        self.history_len = config.history_len
        self.feature_dir = config.feature_dir
        self.feature_types = config.feature_types
        self.downsample = config.downsample
        self.corpus_type = config.corpus_type # da, ms, or asr

        # Other attributes
        self.label_tokenizer = label_tokenizer
        self.tokenizer = tokenizer
        self.pad_label_id = label_tokenizer.pad_token_id
        self.bos_label_id = label_tokenizer.bos_token_id
        self.eos_label_id = label_tokenizer.eos_token_id
        self.statistics = {"n_sessions": 0, 
                "n_uttrs": 0, 
                "n_tokens": 0, 
                "n_segments": 0, 
                "n_fragments": 0}
    
        
    # df columns in files:
    #  filenum  da_speaker  true_speaker  turn_id    da_label  
    #  da_sent  ms_sent     start_times    end_times
    def get_samples(self, split, suffix="_bert_time_data.json"):
        """Creates examples sets."""
        filename = os.path.join(self.dataset_path, split + suffix)
        with open(filename, 'r') as f:
            sessions = json.load(f)

        # Process sessions
        for filenum, sess in sessions.items():
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

                uttr_tokens = [self.tokenizer.cls_token] + uttr_tokens + [self.tokenizer.sep_token]
                uttr_token_ids = self.tokenizer.convert_tokens_to_ids(uttr_tokens)
                uttr_label_ids = [self.bos_label_id] + [self.label_tokenizer.word2id[label] for label in uttr_labels] + [self.eos_label_id]
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

    def load_frames(self, dialog_id: str):
        '''
        param dialog_id: the swbd dialog number, cast to str
        param downsample: whether to skip frames
        '''
        data = {}
        lengths = []
        for speaker in ['A', 'B']:
            data[speaker] = []
            for feat in self.feature_types:
                filepath = os.path.join(self.feature_dir, feat)
                filename = os.path.join(filepath, 'sw' + dialog_id + \
                        '-' + speaker + '.json')
                with open(filename) as fIn:
                    frames = json.load(fIn)
                #print(len(frames))
                if self.downsample:
                    frames = frames[::2]
                lengths.append(len(frames))
                #print(len(frames))
                if not data[speaker]:
                    data[speaker] = frames
                else:
                    # zip keeps everything and truncates at end
                    data[speaker] = [x+y for x,y in zip(data[speaker],frames)]
        return data

    def epoch_init(self, shuffle=False):
        self.cur_fragment_idx = 0
        if shuffle:
            self.fragments = copy.deepcopy(self._fragments)
            random.shuffle(self.fragments)
        else:
            self.fragments = self._fragments

    def __len__(self):
        return len(self._fragments)
    def __len__(self):
        return len(self.idx)

    def get_num_utterances(self):
        return len(self.utterances)

    # load by files, NOT by sentences
    def __getitem__(self, item):
        dialog_id = self.idx[item]
        dialog_info = {}
        dialog_guids = [guid for guid in self.guids \
                if guid.startswith(dialog_id)]
        if self.feature_types is not None:
            data = self.load_frames(dialog_id)
        else:
            data = None
        if self.downsample:
            MULT = 50
        else:
            MULT = 100
        for guid in dialog_guids:
            texts, label, speaker, start, end = self.utterances[guid]
            frame_start = int(np.floor(start*MULT))
            frame_end = int(np.ceil(end*MULT))
            if frame_end - frame_start <= 0:
                print("empty frames for:", guid, frame_end - frame_start)
                # just use the consecutive frames
                frame_end = frame_start + 3
            if self.feature_types is not None:
                frames = data[speaker][frame_start:frame_end]
            else:
                frames = []
            dialog_info[guid] = [texts, frames, label]
        return dialog_info


    def next(self, batch_size):
        # Return None when running out of segments
        if self.cur_fragment_idx == len(self.fragments):
            return None

        # Data to fill in
        X, Y, X_floor = [], [], []

        empty_sent = ""
        empty_tokens = self.tokenizer.convert_string_to_tokens(empty_sent)
        empty_ids = self.tokenizer.convert_tokens_to_ids(empty_tokens)
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
            this_X, this_X_floor = [], []
            for segment in segments:
                this_X.append(segment["token_ids"])
                this_X_floor.append(segment["floor_id"])
            
            segment = segments[-1]
            Y.append(segment["label_ids"])

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
        X_floor = torch.LongTensor(X_floor).to(DEVICE).view(batch_size, self.history_len)
        Y = torch.LongTensor(Y).to(DEVICE).view(batch_size, -1)

        X = X.to(DEVICE) 
        X_type_ids = X_type_ids.to(DEVICE)
        X_attn_masks = X_attn_masks.to(DEVICE)
        batch_data_dict = {
            "X": X,
            "X_floor": X_floor,
            "X_type_ids": X_type_ids,
            "X_attn_masks": X_attn_masks,
            "sent_ids": sent_ids,
            "sent_features": sent_features,
            "Y": Y
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
                for segment in uttr:
                    text = segment["text"]
                    floor = segment["floor"]
                    dialog_act = segment["segment_meta"]["dialog_act"]
                    tokens = self.tokenizer.convert_string_to_tokens(text)[:self.max_uttr_len]
                    uttr_tokens += tokens
                    uttr_labels += ["I"] * (len(tokens) - 1) + ["E_"+dialog_act]

                uttr_tokens = [self.tokenizer.cls_token] + uttr_tokens + [self.tokenizer.sep_token]
                uttr_token_ids = self.tokenizer.convert_tokens_to_ids(uttr_tokens)
                uttr_label_ids = [self.bos_label_id] + [self.label_tokenizer.word2id[label] for label in uttr_labels] + [self.eos_label_id]
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

    # NOTE: for bert-based tokenizers, add pad/cls/attention ids 
    # Similar to sentence_transformers/src/models/BERT.py --> get_sentence_features
    def next(self, batch_size):
        # Return None when running out of segments
        if self.cur_fragment_idx == len(self.fragments):
            return None

        # Data to fill in
        X, Y, X_floor = [], [], []

        empty_sent = ""
        empty_tokens = self.tokenizer.convert_string_to_tokens(empty_sent)
        empty_ids = self.tokenizer.convert_tokens_to_ids(empty_tokens)
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
            this_X, this_X_floor = [], []
            for segment in segments:
                this_X.append(segment["token_ids"])
                this_X_floor.append(segment["floor_id"])
            
            segment = segments[-1]
            Y.append(segment["label_ids"])

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
        X_floor = torch.LongTensor(X_floor).to(DEVICE).view(batch_size, self.history_len)
        Y = torch.LongTensor(Y).to(DEVICE).view(batch_size, -1)

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
            "Y": Y
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
