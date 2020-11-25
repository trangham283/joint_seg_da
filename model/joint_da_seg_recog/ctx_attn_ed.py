import code

import torch
import torch.nn as nn
from torch.nn import functional as F

#from model.joint_da_seg_recog.ctx_ed import BertEdSeqLabeler
from model.modules.encoders import EncoderRNN
from model.modules.decoders import DecoderRNN
from model.modules.submodules import RelFloorEmbEncoder
from model.modules.utils import init_module_weights, init_word_embedding

from transformers import BertModel, BertTokenizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertEmbedder(nn.Module):
    def __init__(self, model_size="bert-base-uncased", 
            cache_dir="/s0/ttmt001",do_lower_case=True, freeze='all'):
        super(BertEmbedder, self).__init__()

        self.bert = BertModel.from_pretrained(model_size, cache_dir=cache_dir)
        self.tokenizer = BertTokenizer.from_pretrained(model_size, 
                do_lower_case=do_lower_case, cache_dir=cache_dir)
        self.embedding_size = self.bert.config.hidden_size
        if freeze == 'pooler_only':
            for name, param in self.bert.named_parameters():
                if "pooler." not in name:
                    param.requires_grad = False
        elif freeze == 'top_layer':
            for name, param in self.bert.named_parameters():
                if "pooler." not in name and "encoder.layer.11." not in name:
                    param.requires_grad = False
        elif freeze == 'all':
            for param in self.bert.parameters(): 
                param.requires_grad = False

        #for name, param in self.bert.named_parameters():
        #    print(name, "size/grad: ", param.size(), param.requires_grad)



    def forward(self, X, X_type_ids, X_attn_masks):
        """Returns token_embeddings"""
        output_tokens = self.bert(input_ids=X, 
                token_type_ids=X_type_ids, 
                attention_mask=X_attn_masks)[0]
        return output_tokens 

class SpeechFeatureEncoder(nn.Module):
    def __init__(self,
            feature_types, feat_sizes, d_out,
            conv_sizes=[5, 10, 25, 50],
            num_conv=32,
            d_pause_embedding=4,  
            pause_vocab_len=9,
            fixed_word_length=100,
            speech_dropout=0.0):
        super().__init__()

        self.d_pause_embedding = d_pause_embedding
        self.d_out = d_out
        self.speech_dropout = nn.Dropout(speech_dropout)
        self.feature_types = feature_types
        self.feat_sizes = feat_sizes
        self.d_in = 0
        self.d_scalars = 0
        self.num_conv = num_conv
        self.conv_sizes = conv_sizes
        self.word_length = fixed_word_length

        all_frame_feats = set(self.feat_sizes.keys())
        frame_feats = all_frame_feats.intersection(set(self.feature_types))
        self.frame_feats = frame_feats

        if 'pause' in self.feature_types:
            self.emb = nn.Embedding(pause_vocab_len, \
                    self.d_pause_embedding)
            self.d_in += 2*self.d_pause_embedding

        if frame_feats:
            conv_modules = []
            feat_dim = 0
            for feat in frame_feats:
                feat_dim += self.feat_sizes[feat]
            for filter_size in self.conv_sizes:
                kernel_size = (filter_size, feat_dim)
                pool_kernel = (self.word_length - filter_size + 1, 1)
                filter_conv = nn.Sequential(
                        nn.Conv2d(1, self.num_conv, kernel_size),
                        nn.ReLU(),
                        nn.MaxPool2d(pool_kernel, 1)
                        )
                conv_modules.append(filter_conv)

            self.conv_modules = nn.ModuleList(conv_modules)

            self.d_conv = self.num_conv * len(self.conv_sizes)
            self.d_in += self.d_conv

        if 'pause_raw' in self.feature_types:
            self.d_scalars += 2
            self.d_in += 2

        if 'word_dur' in self.feature_types:
            self.d_scalars += 2
            self.d_in += 2

        self.speech_projection = nn.Linear(self.d_in, self.d_out, bias=True)

    def forward(self, processed_features):
        pause_features, frame_features, scalar_features = processed_features
        pause_before, pause_after = pause_features
        rp, wd = scalar_features
        all_features = []
        if "pause" in self.feature_types:
            all_features.append(self.emb(pause_before))
            all_features.append(self.emb(pause_after))

        if 'pause_raw' in self.feature_types:
            all_features.append(rp.transpose(1, 2))

        if 'word_dur' in self.feature_types:
            all_features.append(wd.transpose(1, 2))
        
        if self.frame_feats:
            ttff = torch.Tensor(frame_features).to(DEVICE)
            ttff = ttff.transpose(0, 1).unsqueeze(2)
            sp_inputs = list(ttff)
            seq = []
            for word in sp_inputs:
                conv_outputs = [convolve(word) for convolve in self.conv_modules]
                conv_outputs = [x.squeeze(-1).squeeze(-1) for x in conv_outputs]
                conv_outputs = torch.cat(conv_outputs, -1).unsqueeze(0)
                seq.append(conv_outputs)

            sp_out = torch.cat(seq, 0).transpose(0, 1)
            all_features.append(sp_out)

        all_features = torch.cat(all_features, -1)
        res = self.speech_dropout(self.speech_projection(all_features))
        return res

class SpeechAttnEDSeqLabeler(nn.Module):
    def __init__(self, config, tokenizer, label_tokenizer, 
            model_size="bert-base-uncased", 
            cache_dir="/s0/ttmt001", freeze='all'):
        super(SpeechAttnEDSeqLabeler, self).__init__()

        # Attributes
        self.feature_types = config.feature_types
        self.feat_sizes = config.feat_sizes
        self.conv_sizes = config.conv_sizes
        self.num_conv = config.num_conv
        self.d_pause_embedding = config.d_pause_embedding
        self.pause_vocab = config.pause_vocab
        self.d_speech = config.d_speech
        self.fixed_word_length = config.fixed_word_length

        self.dial_encoder_hidden_dim = config.dial_encoder_hidden_dim
        self.n_dial_encoder_layers = config.n_dial_encoder_layers
        self.attention_type = config.attention_type
        self.history_len = config.history_len
        self.num_labels = len(label_tokenizer)
        self.attr_embedding_dim = config.attr_embedding_dim
        self.sent_encoder_hidden_dim = config.sent_encoder_hidden_dim
        self.n_sent_encoder_layers = config.n_sent_encoder_layers
        self.decoder_hidden_dim = config.decoder_hidden_dim
        self.n_decoder_layers = config.n_decoder_layers
        self.decode_max_len = config.decode_max_len
        self.tie_weights = config.tie_weights
        self.rnn_type = config.rnn_type
        self.gen_type = config.gen_type
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.temp = config.temp

        # Optional attributes from config
        self.dropout = config.dropout if hasattr(config, "dropout") else 0.0

        # Bert specific args
        self.word_embedding = BertEmbedder(model_size, cache_dir=cache_dir, 
                freeze=freeze)
        self.word_embedding_dim = self.word_embedding.embedding_size
        
        # Other attributes
        self.word2id = tokenizer.word2id
        self.id2word = tokenizer.id2word
        self.label2id = label_tokenizer.word2id
        self.id2label = label_tokenizer.id2word
        self.vocab_size = len(tokenizer)
        self.label_vocab_size = len(label_tokenizer)
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_label_id = label_tokenizer.pad_token_id
        self.bos_label_id = label_tokenizer.bos_token_id
        self.eos_label_id = label_tokenizer.eos_token_id

        # Encoding components
        self.speech_encoder = SpeechFeatureEncoder(
            self.feature_types, self.feat_sizes,
            self.d_speech, 
            conv_sizes=self.conv_sizes,
            num_conv=self.num_conv,
            d_pause_embedding=self.d_pause_embedding,
            pause_vocab_len=len(self.pause_vocab),
            fixed_word_length=self.fixed_word_length,
            speech_dropout=self.dropout)

        self.sent_encoder = EncoderRNN(
            input_dim=self.word_embedding_dim  + self.d_speech,
            hidden_dim=self.sent_encoder_hidden_dim,
            n_layers=self.n_sent_encoder_layers,
            dropout_emb=self.dropout,
            dropout_input=self.dropout,
            dropout_hidden=self.dropout,
            dropout_output=self.dropout,
            bidirectional=True,
            rnn_type=self.rnn_type,
        )
        self.dial_encoder = EncoderRNN(
            input_dim=self.sent_encoder_hidden_dim,
            hidden_dim=self.dial_encoder_hidden_dim,
            n_layers=self.n_dial_encoder_layers,
            dropout_emb=self.dropout,
            dropout_input=self.dropout,
            dropout_hidden=self.dropout,
            dropout_output=self.dropout,
            bidirectional=False,
            rnn_type=self.rnn_type,
        )

        # Decoding components
        self.enc2dec_hidden_fc = nn.Linear(
            self.dial_encoder_hidden_dim,
            self.n_decoder_layers*self.decoder_hidden_dim if self.rnn_type == "gru"
            else self.n_decoder_layers*self.decoder_hidden_dim*2
        )
        self.label_embedding = nn.Embedding(
            self.label_vocab_size,
            self.attr_embedding_dim,
            padding_idx=self.pad_label_id,
            _weight=init_word_embedding(
                load_pretrained_word_embedding=False,
                id2word=self.id2label,
                word_embedding_dim=self.attr_embedding_dim,
                vocab_size=self.label_vocab_size,
                pad_token_id=self.pad_label_id
            ),
        )
        self.decoder = DecoderRNN(
            vocab_size=self.label_vocab_size,
            input_dim=self.attr_embedding_dim,
            hidden_dim=self.decoder_hidden_dim,
            feat_dim=self.sent_encoder_hidden_dim,
            n_layers=self.n_decoder_layers,
            bos_token_id=self.bos_label_id,
            eos_token_id=self.eos_label_id,
            pad_token_id=self.pad_label_id,
            max_len=self.decode_max_len,
            dropout_emb=self.dropout,
            dropout_input=self.dropout,
            dropout_hidden=self.dropout,
            dropout_output=self.dropout,
            embedding=self.label_embedding,
            tie_weights=self.tie_weights,
            rnn_type=self.rnn_type,
            use_attention=True,
            attn_dim=self.sent_encoder_hidden_dim
        )

        self.floor_encoder = RelFloorEmbEncoder(
            input_dim=self.sent_encoder_hidden_dim,
            embedding_dim=self.attr_embedding_dim
        )

    
    def _init_weights(self):
        init_module_weights(self.enc2dec_hidden_fc)

    def _init_dec_hiddens(self, context):
        batch_size = context.size(0)

        hiddens = self.enc2dec_hidden_fc(context)
        if self.rnn_type == "gru":
            hiddens = hiddens.view(
                batch_size,
                self.n_decoder_layers,
                self.decoder_hidden_dim
            ).transpose(0, 1).contiguous()  # (n_layers, batch_size, hidden_dim)
        elif self.rnn_type == "lstm":
            hiddens = hiddens.view(
                batch_size,
                self.n_decoder_layers,
                self.decoder_hidden_dim,
                2
            )
            # (n_layers, batch_size, hidden_dim)            
            h = hiddens[:, :, :, 0].transpose(0, 1).contiguous()  
            c = hiddens[:, :, :, 1].transpose(0, 1).contiguous()
            hiddens = (h, c)

        return hiddens

    def _get_attn_mask(self, attn_keys):
        attn_mask = (attn_keys != self.pad_token_id)
        return attn_mask

    # FIXME
    def _encode(self, data):
        X_data = data["X"]
        batch_size = X_data.size(0) // self.history_len
        X = X_data.view(batch_size, self.history_len, -1)
        X_type_ids = data["X_type_ids"] 
        X_attn_masks = data["X_attn_masks"]
        input_floors = data["X_floor"]
        Xtext = self.word_embedding(X_data, X_type_ids, X_attn_masks)
        
        Xspeech = self.speech_encoder(data["X_speech"])
        embedded_inputs = torch.cat([Xtext, Xspeech], -1)

        max_sent_len = X.size(-1)

        input_lens = (X != self.pad_token_id).sum(-1)
        dial_lens = (input_lens > 0).long().sum(1)  # equals number of non-padding sents
        flat_input_lens = input_lens.view(batch_size*self.history_len)

        word_encodings, _, sent_encodings = self.sent_encoder(embedded_inputs, 
                flat_input_lens)
        word_encodings = word_encodings.view(batch_size, self.history_len, 
                max_sent_len, -1)
        sent_encodings = sent_encodings.view(batch_size, self.history_len, -1)

        # fetch target-sentence-releated information
        tgt_floors = []
        tgt_word_encodings = []
        for dial_idx, dial_len in enumerate(dial_lens):
            tgt_floors.append(input_floors[dial_idx, dial_len-1])
            tgt_word_encodings.append(word_encodings[dial_idx,dial_len-1, :, :])
        tgt_floors = torch.stack(tgt_floors, 0)
        tgt_word_encodings = torch.stack(tgt_word_encodings, 0)

        src_floors = input_floors.view(-1)
        tgt_floors = tgt_floors.unsqueeze(1).repeat(1,self.history_len).view(-1)
        sent_encodings = sent_encodings.view(batch_size*self.history_len, -1)
        sent_encodings = self.floor_encoder(
            sent_encodings,
            src_floors=src_floors,
            tgt_floors=tgt_floors
        )
        sent_encodings = sent_encodings.view(batch_size, self.history_len, -1)
        
        # [batch_size, dialog_encoder_dim]
        _, _, dial_encodings = self.dial_encoder(sent_encodings, dial_lens)  

        return word_encodings, sent_encodings, dial_encodings, tgt_word_encodings, batch_size

    def _decode(self, dec_inputs, word_encodings, sent_encodings, attn_ctx=None, attn_mask=None):
        batch_size = sent_encodings.size(0)
        hiddens = self._init_dec_hiddens(sent_encodings)
        feats = word_encodings[:, 1:, :].contiguous()  # excluding <s>
        ret_dict = self.decoder.forward(
            batch_size=batch_size,
            inputs=dec_inputs,
            hiddens=hiddens,
            feats=feats,
            attn_ctx=attn_ctx,
            attn_mask=attn_mask,
            mode=DecoderRNN.MODE_TEACHER_FORCE
        )

        return ret_dict

    def _sample(self, word_encodings, sent_encodings, attn_ctx=None, attn_mask=None):
        batch_size = sent_encodings.size(0)
        hiddens = self._init_dec_hiddens(sent_encodings)
        feats = word_encodings[:, 1:, :].contiguous()  # excluding <s>
        ret_dict = self.decoder.forward(
            batch_size=batch_size,
            hiddens=hiddens,
            feats=feats,
            attn_ctx=attn_ctx,
            attn_mask=attn_mask,
            mode=DecoderRNN.MODE_FREE_RUN,
            gen_type=self.gen_type,
            temp=self.temp,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        return ret_dict


    def train_step(self, data):
        # Forward
        word_encodings, sent_encodings, dial_encodings, tgt_word_encodings, batch_size = self._encode(data)
        X_data = data["X"]
        X = X_data.view(batch_size, self.history_len, -1)
        Y = data["Y"]
        Y_in = Y[:, :-1].contiguous()
        Y_out = Y[:, 1:].contiguous()
        max_y_len = Y_out.size(1)

        if self.attention_type == "word":
            attn_keys = word_encodings.view(batch_size, -1, word_encodings.size(-1))
            attn_mask = self._get_attn_mask(X).view(batch_size, -1)
        elif self.attention_type == "sent":
            attn_keys = sent_encodings.view(batch_size, -1, sent_encodings.size(-1))
            attn_mask = (X != self.pad_token_id).sum(-1) > 0
        decoder_ret_dict = self._decode(
            dec_inputs=Y_in,
            word_encodings=tgt_word_encodings,
            sent_encodings=dial_encodings,
            attn_ctx=attn_keys,
            attn_mask=attn_mask
        )

        # Calculate loss
        loss = 0
        logits = decoder_ret_dict["logits"]
        label_losses = F.cross_entropy(
            logits.view(-1, self.label_vocab_size),
            Y_out.view(-1),
            ignore_index=self.pad_label_id,
            reduction="none"
        ).view(batch_size, max_y_len)
        sent_loss = label_losses.sum(1).mean(0)
        loss += sent_loss
        
        # return dicts
        ret_data = {
            "loss": loss
        }
        ret_stat = {
            "loss": loss.item()
        }

        return ret_data, ret_stat

    def evaluate_step(self, data):
        with torch.no_grad():
            # Forward
            word_encodings, sent_encodings, dial_encodings, tgt_word_encodings, batch_size = self._encode(data)
            X_data = data["X"]
            X = X_data.view(batch_size, self.history_len, -1)
            Y = data["Y"]
            Y_in = Y[:, :-1].contiguous()
            Y_out = Y[:, 1:].contiguous()
            max_y_len = Y_out.size(1)

            if self.attention_type == "word":
                attn_keys = word_encodings.view(batch_size, -1, word_encodings.size(-1))
                attn_mask = self._get_attn_mask(X).view(batch_size, -1)
            elif self.attention_type == "sent":
                attn_keys = sent_encodings.view(batch_size, -1, sent_encodings.size(-1))
                attn_mask = (X != self.pad_token_id).sum(-1) > 0
            decoder_ret_dict = self._decode(
                dec_inputs=Y_in,
                word_encodings=tgt_word_encodings,
                sent_encodings=dial_encodings,
                attn_ctx=attn_keys,
                attn_mask=attn_mask
            )

            # Calculate loss
            loss = 0
            logits = decoder_ret_dict["logits"]
            label_losses = F.cross_entropy(
                logits.view(-1, self.label_vocab_size),
                Y_out.view(-1),
                ignore_index=self.pad_label_id,
                reduction="none"
            ).view(batch_size, max_y_len)
            sent_loss = label_losses.sum(1).mean(0)
            loss += sent_loss
        
        # return dicts
        ret_data = {}
        ret_stat = {
            "monitor": loss.item(),
            "loss": loss.item()
        }

        return ret_data, ret_stat

    def test_step(self, data):
        with torch.no_grad():
            # Forward
            word_encodings, sent_encodings, dial_encodings, tgt_word_encodings, batch_size = self._encode(data)
            X_data = data["X"]
            X = X_data.view(batch_size, self.history_len, -1)

            if self.attention_type == "word":
                attn_keys = word_encodings.view(batch_size, -1, word_encodings.size(-1))
                attn_mask = self._get_attn_mask(X).view(batch_size, -1)
            elif self.attention_type == "sent":
                attn_keys = sent_encodings.view(batch_size, -1, sent_encodings.size(-1))
                attn_mask = (X != self.pad_token_id).sum(-1) > 0
            decoder_ret_dict = self._sample(
                word_encodings=tgt_word_encodings,
                sent_encodings=dial_encodings,
                attn_ctx=attn_keys,
                attn_mask=attn_mask
            )

        ret_data = {
            "symbols": decoder_ret_dict["symbols"]
        }
        ret_stat = {}

        return ret_data, ret_stat

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


class BertAttnEDSeqLabeler(nn.Module):
    def __init__(self, config, tokenizer, label_tokenizer, 
            model_size="bert-base-uncased", 
            cache_dir="/s0/ttmt001", freeze='all'):
        super(BertAttnEDSeqLabeler, self).__init__()

        # Attributes
        # Attributes from config
        self.dial_encoder_hidden_dim = config.dial_encoder_hidden_dim
        self.n_dial_encoder_layers = config.n_dial_encoder_layers
        self.attention_type = config.attention_type
        self.history_len = config.history_len
        self.num_labels = len(label_tokenizer)
        self.attr_embedding_dim = config.attr_embedding_dim
        self.sent_encoder_hidden_dim = config.sent_encoder_hidden_dim
        self.n_sent_encoder_layers = config.n_sent_encoder_layers
        self.decoder_hidden_dim = config.decoder_hidden_dim
        self.n_decoder_layers = config.n_decoder_layers
        self.decode_max_len = config.decode_max_len
        self.tie_weights = config.tie_weights
        self.rnn_type = config.rnn_type
        self.gen_type = config.gen_type
        self.top_k = config.top_k
        self.top_p = config.top_p
        self.temp = config.temp

        # Optional attributes from config
        self.dropout = config.dropout if hasattr(config, "dropout") else 0.0
        self.use_pretrained_word_embedding = config.use_pretrained_word_embedding if hasattr(config, "use_pretrained_word_embedding") else False

        # Bert specific args
        self.word_embedding = BertEmbedder(model_size, cache_dir=cache_dir, 
                freeze=freeze)
        self.word_embedding_dim = self.word_embedding.embedding_size
        
        # Other attributes
        self.word2id = tokenizer.word2id
        self.id2word = tokenizer.id2word
        self.label2id = label_tokenizer.word2id
        self.id2label = label_tokenizer.id2word
        self.vocab_size = len(tokenizer)
        self.label_vocab_size = len(label_tokenizer)
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_label_id = label_tokenizer.pad_token_id
        self.bos_label_id = label_tokenizer.bos_token_id
        self.eos_label_id = label_tokenizer.eos_token_id

        # Encoding components
        self.sent_encoder = EncoderRNN(
            input_dim=self.word_embedding_dim,
            hidden_dim=self.sent_encoder_hidden_dim,
            n_layers=self.n_sent_encoder_layers,
            dropout_emb=self.dropout,
            dropout_input=self.dropout,
            dropout_hidden=self.dropout,
            dropout_output=self.dropout,
            bidirectional=True,
            rnn_type=self.rnn_type,
        )
        self.dial_encoder = EncoderRNN(
            input_dim=self.sent_encoder_hidden_dim,
            hidden_dim=self.dial_encoder_hidden_dim,
            n_layers=self.n_dial_encoder_layers,
            dropout_emb=self.dropout,
            dropout_input=self.dropout,
            dropout_hidden=self.dropout,
            dropout_output=self.dropout,
            bidirectional=False,
            rnn_type=self.rnn_type,
        )

        # Decoding components
        self.enc2dec_hidden_fc = nn.Linear(
            self.dial_encoder_hidden_dim,
            self.n_decoder_layers*self.decoder_hidden_dim if self.rnn_type == "gru"
            else self.n_decoder_layers*self.decoder_hidden_dim*2
        )
        self.label_embedding = nn.Embedding(
            self.label_vocab_size,
            self.attr_embedding_dim,
            padding_idx=self.pad_label_id,
            _weight=init_word_embedding(
                load_pretrained_word_embedding=False,
                id2word=self.id2label,
                word_embedding_dim=self.attr_embedding_dim,
                vocab_size=self.label_vocab_size,
                pad_token_id=self.pad_label_id
            ),
        )
        self.decoder = DecoderRNN(
            vocab_size=self.label_vocab_size,
            input_dim=self.attr_embedding_dim,
            hidden_dim=self.decoder_hidden_dim,
            feat_dim=self.sent_encoder_hidden_dim,
            n_layers=self.n_decoder_layers,
            bos_token_id=self.bos_label_id,
            eos_token_id=self.eos_label_id,
            pad_token_id=self.pad_label_id,
            max_len=self.decode_max_len,
            dropout_emb=self.dropout,
            dropout_input=self.dropout,
            dropout_hidden=self.dropout,
            dropout_output=self.dropout,
            embedding=self.label_embedding,
            tie_weights=self.tie_weights,
            rnn_type=self.rnn_type,
            use_attention=True,
            attn_dim=self.sent_encoder_hidden_dim
        )

        self.floor_encoder = RelFloorEmbEncoder(
            input_dim=self.sent_encoder_hidden_dim,
            embedding_dim=self.attr_embedding_dim
        )
    
    def _init_weights(self):
        init_module_weights(self.enc2dec_hidden_fc)

    def _init_dec_hiddens(self, context):
        batch_size = context.size(0)

        hiddens = self.enc2dec_hidden_fc(context)
        if self.rnn_type == "gru":
            hiddens = hiddens.view(
                batch_size,
                self.n_decoder_layers,
                self.decoder_hidden_dim
            ).transpose(0, 1).contiguous()  # (n_layers, batch_size, hidden_dim)
        elif self.rnn_type == "lstm":
            hiddens = hiddens.view(
                batch_size,
                self.n_decoder_layers,
                self.decoder_hidden_dim,
                2
            )
            # (n_layers, batch_size, hidden_dim)            
            h = hiddens[:, :, :, 0].transpose(0, 1).contiguous()  
            c = hiddens[:, :, :, 1].transpose(0, 1).contiguous()
            hiddens = (h, c)

        return hiddens

    def _get_attn_mask(self, attn_keys):
        attn_mask = (attn_keys != self.pad_token_id)
        return attn_mask

    def _encode(self, inputs, input_floors, embedded_inputs):
        batch_size, history_len, max_sent_len = inputs.size()

        input_lens = (inputs != self.pad_token_id).sum(-1)
        dial_lens = (input_lens > 0).long().sum(1)  # equals number of non-padding sents
        flat_inputs = inputs.view(batch_size*history_len, max_sent_len)
        flat_input_lens = input_lens.view(batch_size*history_len)

        word_encodings, _, sent_encodings = self.sent_encoder(embedded_inputs, 
                flat_input_lens)
        word_encodings = word_encodings.view(batch_size, self.history_len, 
                max_sent_len, -1)
        sent_encodings = sent_encodings.view(batch_size, self.history_len, -1)

        # fetch target-sentence-releated information
        tgt_floors = []
        tgt_word_encodings = []
        for dial_idx, dial_len in enumerate(dial_lens):
            tgt_floors.append(input_floors[dial_idx, dial_len-1])
            tgt_word_encodings.append(word_encodings[dial_idx,dial_len-1, :, :])
        tgt_floors = torch.stack(tgt_floors, 0)
        tgt_word_encodings = torch.stack(tgt_word_encodings, 0)

        src_floors = input_floors.view(-1)
        tgt_floors = tgt_floors.unsqueeze(1).repeat(1,self.history_len).view(-1)
        sent_encodings = sent_encodings.view(batch_size*self.history_len, -1)
        sent_encodings = self.floor_encoder(
            sent_encodings,
            src_floors=src_floors,
            tgt_floors=tgt_floors
        )
        sent_encodings = sent_encodings.view(batch_size, self.history_len, -1)
        
        # [batch_size, dialog_encoder_dim]
        _, _, dial_encodings = self.dial_encoder(sent_encodings, dial_lens)  

        return word_encodings, sent_encodings, dial_encodings, tgt_word_encodings

    def _decode(self, dec_inputs, word_encodings, sent_encodings, attn_ctx=None, attn_mask=None):
        batch_size = sent_encodings.size(0)
        hiddens = self._init_dec_hiddens(sent_encodings)
        feats = word_encodings[:, 1:, :].contiguous()  # excluding <s>
        ret_dict = self.decoder.forward(
            batch_size=batch_size,
            inputs=dec_inputs,
            hiddens=hiddens,
            feats=feats,
            attn_ctx=attn_ctx,
            attn_mask=attn_mask,
            mode=DecoderRNN.MODE_TEACHER_FORCE
        )

        return ret_dict

    def _sample(self, word_encodings, sent_encodings, attn_ctx=None, attn_mask=None):
        batch_size = sent_encodings.size(0)
        hiddens = self._init_dec_hiddens(sent_encodings)
        feats = word_encodings[:, 1:, :].contiguous()  # excluding <s>
        ret_dict = self.decoder.forward(
            batch_size=batch_size,
            hiddens=hiddens,
            feats=feats,
            attn_ctx=attn_ctx,
            attn_mask=attn_mask,
            mode=DecoderRNN.MODE_FREE_RUN,
            gen_type=self.gen_type,
            temp=self.temp,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        return ret_dict


    def train_step(self, data):
        """One training step

        NOTE: Arguments REFERENCE ONLY:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of sentences
                'Y' {LongTensor [batch_size, max_sent_len]} -- label ids of corresponding tokens

        Returns:
            dict of data -- returned keys and values
                'loss' {FloatTensor []} -- loss to backward
            dict of statistics -- returned keys and values
                'loss' {float} -- batch loss
        """
        X_data, Y = data["X"], data["Y"]
        batch_size = X_data.size(0) // self.history_len
        X = X_data.view(batch_size, self.history_len, -1)
        X_type_ids, X_attn_masks = data["X_type_ids"], data["X_attn_masks"]
        X1 = self.word_embedding(X_data, X_type_ids, X_attn_masks)
        X_floor = data["X_floor"]
        Y_in = Y[:, :-1].contiguous()
        Y_out = Y[:, 1:].contiguous()
        max_y_len = Y_out.size(1)

        # Forward
        word_encodings, sent_encodings, dial_encodings, tgt_word_encodings = self._encode(X, X_floor, X1)
        if self.attention_type == "word":
            attn_keys = word_encodings.view(batch_size, -1, word_encodings.size(-1))
            attn_mask = self._get_attn_mask(X).view(batch_size, -1)
        elif self.attention_type == "sent":
            attn_keys = sent_encodings.view(batch_size, -1, sent_encodings.size(-1))
            attn_mask = (X != self.pad_token_id).sum(-1) > 0
        decoder_ret_dict = self._decode(
            dec_inputs=Y_in,
            word_encodings=tgt_word_encodings,
            sent_encodings=dial_encodings,
            attn_ctx=attn_keys,
            attn_mask=attn_mask
        )

        # Calculate loss
        loss = 0
        logits = decoder_ret_dict["logits"]
        label_losses = F.cross_entropy(
            logits.view(-1, self.label_vocab_size),
            Y_out.view(-1),
            ignore_index=self.pad_label_id,
            reduction="none"
        ).view(batch_size, max_y_len)
        sent_loss = label_losses.sum(1).mean(0)
        loss += sent_loss
        
        # return dicts
        ret_data = {
            "loss": loss
        }
        ret_stat = {
            "loss": loss.item()
        }

        return ret_data, ret_stat

    def evaluate_step(self, data):
        """One evaluation step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of sentences
                'Y' {LongTensor [batch_size, 1, max_sent_len]} -- label ids of corresponding tokens

        Returns:
            dict of data -- returned keys and values

            dict of statistics -- returned keys and values
                'loss' {float} -- batch loss
        """
        X_data, Y = data["X"], data["Y"]
        batch_size = X_data.size(0) // self.history_len
        X = X_data.view(batch_size, self.history_len, -1)
        X_type_ids, X_attn_masks = data["X_type_ids"], data["X_attn_masks"]
        X1 = self.word_embedding(X_data, X_type_ids, X_attn_masks)
        X_floor = data["X_floor"]
        Y_in = Y[:, :-1].contiguous()
        Y_out = Y[:, 1:].contiguous()
        max_y_len = Y_out.size(1)
        
        with torch.no_grad():
            # Forward
            word_encodings, sent_encodings, dial_encodings, tgt_word_encodings = self._encode(X, X_floor, X1)
            if self.attention_type == "word":
                attn_keys = word_encodings.view(batch_size, -1, word_encodings.size(-1))
                attn_mask = self._get_attn_mask(X).view(batch_size, -1)
            elif self.attention_type == "sent":
                attn_keys = sent_encodings.view(batch_size, -1, sent_encodings.size(-1))
                attn_mask = (X != self.pad_token_id).sum(-1) > 0
            decoder_ret_dict = self._decode(
                dec_inputs=Y_in,
                word_encodings=tgt_word_encodings,
                sent_encodings=dial_encodings,
                attn_ctx=attn_keys,
                attn_mask=attn_mask
            )

            # Calculate loss
            loss = 0
            logits = decoder_ret_dict["logits"]
            label_losses = F.cross_entropy(
                logits.view(-1, self.label_vocab_size),
                Y_out.view(-1),
                ignore_index=self.pad_label_id,
                reduction="none"
            ).view(batch_size, max_y_len)
            sent_loss = label_losses.sum(1).mean(0)
            loss += sent_loss
        
        # return dicts
        ret_data = {}
        ret_stat = {
            "monitor": loss.item(),
            "loss": loss.item()
        }

        return ret_data, ret_stat

    def test_step(self, data):
        """One test step

        Arguments:
            data {dict of data} -- required keys and values:
                'X' {LongTensor [batch_size, history_len, max_x_sent_len]} -- token ids of sentences
                'X_floor' {LongTensor [batch_size, history_len]} -- floors of sentences

        Returns:
            dict of data -- returned keys and values
                'symbols' {LongTensor [batch_size, max_decode_len]} -- predicted label ids
            dict of statistics -- returned keys and values
        """
        X_data = data["X"]
        batch_size = X_data.size(0) // self.history_len
        X = X_data.view(batch_size, self.history_len, -1)
        X_type_ids, X_attn_masks = data["X_type_ids"], data["X_attn_masks"]
        X1 = self.word_embedding(X_data, X_type_ids, X_attn_masks)
        X_floor = data["X_floor"]
        
        with torch.no_grad():
            # Forward
            word_encodings, sent_encodings, dial_encodings, tgt_word_encodings = self._encode(X, X_floor, X1)
            if self.attention_type == "word":
                attn_keys = word_encodings.view(batch_size, -1, word_encodings.size(-1))
                attn_mask = self._get_attn_mask(X).view(batch_size, -1)
            elif self.attention_type == "sent":
                attn_keys = sent_encodings.view(batch_size, -1, sent_encodings.size(-1))
                attn_mask = (X != self.pad_token_id).sum(-1) > 0
            decoder_ret_dict = self._sample(
                word_encodings=tgt_word_encodings,
                sent_encodings=dial_encodings,
                attn_ctx=attn_keys,
                attn_mask=attn_mask
            )

        ret_data = {
            "symbols": decoder_ret_dict["symbols"]
        }
        ret_stat = {}

        return ret_data, ret_stat

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

