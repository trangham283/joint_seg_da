import re
import code
import math
import json
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.modules.utils import init_module_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Roberta(nn.Module):
    def __init__(self, config, tokenizer):
        super(Roberta, self).__init__()

        # Load pretrained gpt2 model
        model_size = config.model_size
        assert model_size in ["base", "large", "large-mnli"]
        from transformers import RobertaForSequenceClassification
        pretrained = RobertaForSequenceClassification.from_pretrained(f'roberta-{model_size}')
        pretrained.resize_token_embeddings(len(tokenizer))
        self.roberta = pretrained.roberta

        # Attributes
        # Attributes from config
        self.num_labels = len(config.dialog_acts)
        # Other attributes
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        self.sep_token_id = tokenizer.sep_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.speaker1_token_id = tokenizer.speaker1_token_id
        self.speaker2_token_id = tokenizer.speaker2_token_id
        self.hidden_dim = self.roberta.config.hidden_size
        self.hidden_dropout_prob = self.roberta.config.hidden_dropout_prob

        # Embedding componenets
        self.embeddings = self.roberta.embeddings

        # Output layer
        self.classifier_pipe = nn.ModuleList(
            [
                nn.Dropout(self.hidden_dropout_prob),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Dropout(self.hidden_dropout_prob),
                nn.Linear(self.hidden_dim, self.num_labels),
            ]
        )

        # Initialization
        self._init_weights()

    def _init_weights(self):
        init_module_weights(self.classifier_pipe)

    def _construct_input_output(self, inputs, input_floors, output_floors):
        """Convert a list of sentences into a single sequence for each dialog
        """

        # Use lists instead of tensors to speed up
        input_lens = (inputs != self.pad_token_id).sum(-1)
        dial_lens = (input_lens > 0).sum(dim=1).tolist()
        inputs = inputs.tolist()
        input_lens = input_lens.tolist()
        input_floors = input_floors.tolist()
        output_floors = output_floors.tolist()
       
        # build sequences
        input_token_id_seqs = []
        for dial_idx in range(len(inputs)):
            # merge history sentences in context (sentences except for the last sentence)
            ctx_input_ids = []
            for sent_idx in range(dial_lens[dial_idx]-1):
                sent_len = input_lens[dial_idx][sent_idx]
                sent_token_ids = inputs[dial_idx][sent_idx][:sent_len]

                src_speaker = input_floors[dial_idx][sent_idx]
                tgt_speaker = output_floors[dial_idx]
                speaker_token_id = self.speaker1_token_id if src_speaker == tgt_speaker else self.speaker2_token_id

                ctx_input_ids += ([speaker_token_id] + sent_token_ids)

            # response token ids (the last sentence)
            sent_idx = dial_lens[dial_idx]-1
            sent_len = input_lens[dial_idx][sent_idx]
            sent_token_ids = inputs[dial_idx][sent_idx][:sent_len]
            response_input_ids = [self.speaker1_token_id] + sent_token_ids

            # concat context and response
            input_token_id_seq = [self.cls_token_id] + ctx_input_ids + [self.sep_token_id] + [self.sep_token_id] + response_input_ids + [self.sep_token_id]
            input_token_id_seqs.append(input_token_id_seq)

        # Get lengths
        seq_lens = [len(seq) for seq in input_token_id_seqs]
        max_seq_len = max(seq_lens)

        # Get attention masks
        attention_masks = [[1]*len(seq) + [0]*(max_seq_len-len(seq)) for seq in input_token_id_seqs]

        # Pad sequences and produce tensors
        input_token_id_seqs = [seq + [self.pad_token_id]*(max_seq_len-len(seq)) for seq in input_token_id_seqs]
        input_token_id_seqs = torch.LongTensor(input_token_id_seqs).to(DEVICE)
        attention_masks = torch.LongTensor(attention_masks).to(DEVICE)

        return {
            "input_ids": input_token_id_seqs,
            "seq_lens": seq_lens,
            "attention_masks": attention_masks,
        }

    def _forward(self, input_ids, attention_masks):
        outputs = self.roberta(input_ids, attention_mask=attention_masks)
        sequence_output = outputs[0]

        cls_hidden = sequence_output[:, 0, :]
        x = cls_hidden
        for m in self.classifier_pipe:
            x = m(x)
        logits = x

        return logits

    def load_model(self, model_path):
        """Load pretrained model weights from model_path

        Arguments:
            model_path {str} -- path to pretrained model weights
        """
        pretrained_state_dict = torch.load(
            model_path,
            map_location=lambda storage, loc: storage
        )
        self.load_state_dict(pretrained_state_dict)

    def train_step(self, data):
        """One training step

        Arguments:
                'X' {LongTensor [batch_size, history_len+1, max_x_sent_len]} -- token ids of context and target sentences
                'X_floor' {LongTensor [batch_size, history_len+1]} -- floors of context and target sentences
                'Y_floor' {LongTensor [batch_size]} -- floor of target sentence
                'Y_da' {LongTensor [batch_size]} -- dialog acts of target sentence

        Returns:
            dict of data -- returned keys and values
                'loss' {FloatTensor []} -- loss tensor to backward
            dict of statistics -- returned keys and values
                'loss' {float} -- batch loss
        """
        X = data["X"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_da = data["Y_da"]

        # Construct inputs and outputs
        data_dict = self._construct_input_output(
            inputs=X,
            input_floors=X_floor,
            output_floors=Y_floor,
        )
        input_ids = data_dict["input_ids"]
        attention_masks = data_dict["attention_masks"]

        # Forward
        logits = self._forward(input_ids, attention_masks)

        # Calculate loss
        loss = F.cross_entropy(
            logits.view(-1, self.num_labels),
            Y_da.view(-1),
            reduction="mean"
        )

        # Return dicts
        ret_data = {
            "loss": loss,
        }
        ret_stat = {
            "loss": loss.item()
        }

        return ret_data, ret_stat

    def evaluate_step(self, data):
        """One evaluation step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len+1, max_x_sent_len]} -- token ids of context and target sentences
                'X_floor' {LongTensor [batch_size, history_len+1]} -- floors of context and target sentences
                'Y_floor' {LongTensor [batch_size]} -- floor of target sentence
                'Y_da' {LongTensor [batch_size]} -- dialog acts of target sentence

        Returns:
            dict of outputs -- returned keys and values
                labels {LongTensor [batch_size]} -- predicted label of target sentence
            dict of statistics -- returned keys and values
                'monitor' {float} -- a monitor number for learning rate scheduling
        """
        X = data["X"]
        X_floor, Y_floor = data["X_floor"], data["Y_floor"]
        Y_da = data["Y_da"]

        with torch.no_grad():
            # Construct inputs and outputs
            data_dict = self._construct_input_output(
                inputs=X,
                input_floors=X_floor,
                output_floors=Y_floor,
            )
            input_ids = data_dict["input_ids"]
            attention_masks = data_dict["attention_masks"]

            # Forward
            logits = self._forward(input_ids, attention_masks)
            _, labels = torch.max(logits, dim=1)

            # Calculate loss
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                Y_da.view(-1),
                reduction="mean"
            )

        # return dicts
        ret_data = {
            "labels": labels
        }
        ret_stat = {
            "monitor": loss.item()
        }

        return ret_data, ret_stat
