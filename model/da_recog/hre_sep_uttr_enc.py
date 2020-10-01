import re
import code
import math
import json
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from model.modules.encoders import EncoderRNN
from model.modules.utils import init_module_weights, init_word_embedding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HRESepUttrEnc(nn.Module):
    def __init__(self, config, tokenizer):
        super(HRESepUttrEnc, self).__init__()

        # Attributes
        # Attributes from config
        self.num_labels = len(config.dialog_acts)
        self.word_embedding_dim = config.word_embedding_dim
        self.attr_embedding_dim = config.attr_embedding_dim
        self.sent_encoder_hidden_dim = config.sent_encoder_hidden_dim
        self.n_sent_encoder_layers = config.n_sent_encoder_layers
        self.dial_encoder_hidden_dim = config.dial_encoder_hidden_dim
        self.n_dial_encoder_layers = config.n_dial_encoder_layers
        self.rnn_type = config.rnn_type
        self.word_embedding_path = config.word_embedding_path
        self.floor_encoder_type = config.floor_encoder
        # Optional attributes from config
        self.dropout_emb = config.dropout if hasattr(config, "dropout") else 0.0
        self.dropout_input = config.dropout if hasattr(config, "dropout") else 0.0
        self.dropout_hidden = config.dropout if hasattr(config, "dropout") else 0.0
        self.dropout_output = config.dropout if hasattr(config, "dropout") else 0.0
        self.use_pretrained_word_embedding = config.use_pretrained_word_embedding if hasattr(config, "use_pretrained_word_embedding") else False
        # Other attributes
        self.word2id = tokenizer.word2id
        self.id2word = tokenizer.id2word
        self.vocab_size = len(tokenizer.word2id)
        self.pad_token_id = tokenizer.pad_token_id

        # Input components
        self.word_embedding = nn.Embedding(
            self.vocab_size,
            self.word_embedding_dim,
            padding_idx=self.pad_token_id,
            _weight=init_word_embedding(
                load_pretrained_word_embedding=self.use_pretrained_word_embedding,
                pretrained_word_embedding_path=self.word_embedding_path,
                id2word=self.id2word,
                word_embedding_dim=self.word_embedding_dim,
                vocab_size=self.vocab_size,
                pad_token_id=self.pad_token_id
            ),
        )

        # Encoding components
        self.own_sent_encoder = EncoderRNN(
            input_dim=self.word_embedding_dim,
            hidden_dim=self.sent_encoder_hidden_dim,
            n_layers=self.n_sent_encoder_layers,
            dropout_emb=self.dropout_emb,
            dropout_input=self.dropout_input,
            dropout_hidden=self.dropout_hidden,
            dropout_output=self.dropout_output,
            bidirectional=True,
            embedding=self.word_embedding,
            rnn_type=self.rnn_type,
        )
        self.oth_sent_encoder = EncoderRNN(
            input_dim=self.word_embedding_dim,
            hidden_dim=self.sent_encoder_hidden_dim,
            n_layers=self.n_sent_encoder_layers,
            dropout_emb=self.dropout_emb,
            dropout_input=self.dropout_input,
            dropout_hidden=self.dropout_hidden,
            dropout_output=self.dropout_output,
            bidirectional=True,
            embedding=self.word_embedding,
            rnn_type=self.rnn_type,
        )
        self.dial_encoder = EncoderRNN(
            input_dim=self.sent_encoder_hidden_dim,
            hidden_dim=self.dial_encoder_hidden_dim,
            n_layers=self.n_dial_encoder_layers,
            dropout_emb=self.dropout_emb,
            dropout_input=self.dropout_input,
            dropout_hidden=self.dropout_hidden,
            dropout_output=self.dropout_output,
            bidirectional=False,
            rnn_type=self.rnn_type,
        )

        # Classification components
        self.output_fc = nn.Linear(
            self.dial_encoder_hidden_dim,
            self.num_labels
        )

        # Initialization
        self._init_weights()

    def _init_weights(self):
        init_module_weights(self.output_fc)

    def _encode(self, inputs, input_floors, output_floors):
        batch_size, history_len, max_x_sent_len = inputs.size()

        # own and others' sentence encodings
        flat_inputs = inputs.view(batch_size*history_len, max_x_sent_len)
        input_lens = (inputs != self.pad_token_id).sum(-1)
        flat_input_lens = input_lens.view(batch_size*history_len)
        own_word_encodings, _, own_sent_encodings = self.own_sent_encoder(flat_inputs, flat_input_lens)
        oth_word_encodings, _, oth_sent_encodings = self.oth_sent_encoder(flat_inputs, flat_input_lens)

        # floor identity flags
        src_floors = input_floors.view(-1)  # batch_size*history_len
        target_floors = output_floors.unsqueeze(1).repeat(1, history_len).view(-1)  # batch_size*history_len
        if self.floor_encoder_type == "abs":
            is_own_uttr_flag = src_floors.bool()
        elif self.floor_encoder_type == "rel":
            is_own_uttr_flag = (target_floors == src_floors)  # batch_size*history_len
        else:
            raise Exception("wrong floor encoder type")

        # gather sent encodings
        stacked_sent_encodings = torch.stack([own_sent_encodings, oth_sent_encodings], dim=1)
        select_mask = torch.stack([is_own_uttr_flag, ~is_own_uttr_flag], dim=1)
        select_mask = select_mask.unsqueeze(2).repeat(1, 1, stacked_sent_encodings.size(-1))
        selected_sent_encodings = torch.masked_select(stacked_sent_encodings, select_mask).view(batch_size, history_len, -1)

        # gather word encodings
        stacked_word_encodings = torch.stack([own_word_encodings, oth_word_encodings], dim=1)
        select_mask = torch.stack([is_own_uttr_flag, ~is_own_uttr_flag], dim=1)
        select_mask = select_mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, max_x_sent_len, stacked_word_encodings.size(-1))
        selected_word_encodings = torch.masked_select(stacked_word_encodings, select_mask).view(batch_size, history_len, max_x_sent_len, -1)

        dialog_lens = (input_lens > 0).long().sum(1)  # equals number of non-padding sents
        _, _, dialog_encodings = self.dial_encoder(selected_sent_encodings, dialog_lens)  # [batch_size, dialog_encoder_dim]

        return selected_word_encodings, selected_sent_encodings, dialog_encodings

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
            data {dict of data} -- required keys and values:
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

        # Forward
        word_encodings, sent_encodings, dial_encodings = self._encode(
            inputs=X,
            input_floors=X_floor,
            output_floors=Y_floor
        )
        logits = self.output_fc(dial_encodings)

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
            # Forward
            word_encodings, sent_encodings, dial_encodings = self._encode(
                inputs=X,
                input_floors=X_floor,
                output_floors=Y_floor
            )
            logits = self.output_fc(dial_encodings)
            _, labels = torch.max(logits, dim=1)

            # Loss
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
