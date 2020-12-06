import code

import torch, math
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import MultiheadAttention
import torch.nn.init as init

from model.modules.encoders import EncoderRNN
from model.modules.decoders import DecoderRNN
from model.modules.submodules import RelFloorEmbEncoder
from model.modules.utils import init_module_weights, init_word_embedding

from transformers import BertModel, BertTokenizer

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch_t = torch.cuda
else:
    DEVICE = torch.device("cpu")
    torch_t = torch


class RelSpeakerEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hist_len):
        super(RelSpeakerEncoder, self).__init__()
        self.input_dim = input_dim
        self.hist_len = hist_len
        self.embedding = nn.Embedding(2, embedding_dim)
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(input_dim+hist_len*embedding_dim, input_dim)
        self.init_weights()

    def init_weights(self):
        init_module_weights(self.embedding, 1.0)
        init_module_weights(self.linear, 0.1)

    def forward(self, word_encodings, src_floors, tgt_floors):
        batch_size, seq_len, hidden_dim = word_encodings.size()
        same_floors = (src_floors == tgt_floors).long()
        floor_embeddings = self.embedding(same_floors)
        floor_embeddings = floor_embeddings.reshape(batch_size, 
                self.hist_len * self.embedding_dim)
        floor_embeddings = floor_embeddings.unsqueeze(1).expand(-1, seq_len, -1)

        encodings = torch.cat([word_encodings, floor_embeddings], dim=-1)
        outputs = self.linear(encodings)

        return outputs


class BertEmbedder(nn.Module):
    def __init__(self, model_size="bert-base-uncased", 
            cache_dir="/s0/ttmt001",do_lower_case=True, freeze='all'):
        super(BertEmbedder, self).__init__()

        self.bert = BertModel.from_pretrained(model_size, cache_dir=cache_dir,
                add_pooling_layer=False)
        self.tokenizer = BertTokenizer.from_pretrained(model_size, 
                do_lower_case=do_lower_case, cache_dir=cache_dir)

        self.embedding_size = self.bert.config.hidden_size
        if freeze == 'top_layer':
            for name, param in self.bert.named_parameters():
                if "pooler." not in name and "encoder.layer.11." not in name:
                    param.requires_grad = False
        elif freeze == 'all':
            for param in self.bert.parameters(): 
                param.requires_grad = False
        # else: tune everything; i.e. freeze=none

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
            init_module_weights(self.emb, 1.0)

        if frame_feats:
            conv_modules = []
            feat_dim = 0
            for feat in frame_feats:
                feat_dim += self.feat_sizes[feat]
            self.feat_dim = feat_dim

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
            init_module_weights(self.conv_modules)

            self.d_conv = self.num_conv * len(self.conv_sizes)
            self.d_in += self.d_conv

        if 'pause_raw' in self.feature_types:
            self.d_scalars += 2
            self.d_in += 2

        if 'word_dur' in self.feature_types:
            self.d_scalars += 2
            self.d_in += 2

        self.speech_projection = nn.Linear(self.d_in, self.d_out, bias=True)
        init_module_weights(self.speech_projection, 0.1)

    def forward(self, processed_features):
        pause_features, frame_features, scalar_features = processed_features
        if pause_features:
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
       
        # Documenting all this dimension finessing is probably futile,
        # and it's not like I'd trust myself enough not to test things out 
        # line by line anyway...
        if frame_features:
            ttff = torch.Tensor(frame_features).to(DEVICE)
            
            batch_size = ttff.size(0)
            seq_len = ttff.size(1)
            inputs = ttff.view(batch_size*seq_len, 
                    self.word_length, self.feat_dim).unsqueeze(1)
            conv_outputs = [convolve(inputs) for convolve in self.conv_modules]
            conv_outputs = [x.squeeze(-1).squeeze(-1) for x in conv_outputs]
            conv_outputs = [x.view(batch_size, seq_len, -1) 
                    for x in conv_outputs]
            sp_out = torch.cat(conv_outputs, -1)
            all_features.append(sp_out)
            
        if not all_features:
            return None
        all_features = torch.cat(all_features, -1)
        res = self.speech_dropout(self.speech_projection(all_features))
        return res

# From repo:
# https://github.com/UKPLab/sentence-transformers
class Pooling(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.
    Using pooling, it generates from a variable sized sentence a fixed sized 
    sentence embedding. This layer also allows to use the CLS token if it is 
    returned by the underlying word embedding model.
    You can concatenate multiple poolings together.
    """
    def __init__(self,
                 input_embedding_dimension: int,
                 output_dim: int,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 ):
        super(Pooling, self).__init__()

        self.config_keys = [
                'input_embedding_dimension', 
                'pooling_mode_cls_token', 
                'pooling_mode_mean_tokens', 
                'pooling_mode_max_tokens', 
                ]

        self.input_embedding_dimension = input_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.output_dim = output_dim

        pooling_mode_multiplier = sum([
            pooling_mode_cls_token, 
            pooling_mode_max_tokens, 
            pooling_mode_mean_tokens])
        self.pooling_output_dimension = pooling_mode_multiplier * input_embedding_dimension
        self.project = nn.Linear(self.pooling_output_dimension, self.output_dim)
        init_module_weights(self.project, 0.1)

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def forward(self, token_embeddings, input_mask):
        cls_token = token_embeddings[:, 0, :] # CLS token is first token

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            # Set padding tokens to large negative value
            token_embeddings[input_mask_expanded == 0] = -1e9  
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens:
            input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)

        output_vector = torch.cat(output_vectors, 1)
        return self.project(output_vector)

# From repo:
# https://github.com/pytorch/examples/blob/master/word_language_model/
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position 
        of the tokens in the sequence. 
        The positional encodings have the same dimension as the embeddings, 
        so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, mode='absolute', comb='cat', dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.mode = mode
        self.comb = comb
        self.d_model = d_model
        self.max_len = max_len

        if self.mode == 'relative':
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            # don't transpose here or unsqueeze
            #pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)
        else:
            self.position_table = nn.Parameter(
                    torch.FloatTensor(max_len, d_model))
            init.normal_(self.position_table)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, seq len, embed dim]
            output: [batch size, seq len, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        if self.mode == 'relative':
            timing_signal = self.pe[:x.size(1), :].unsqueeze(0)
        else:
            timing_signal = self.position_table[:x.size(1), :].unsqueeze(0)

        if self.comb == 'add':
            x = x + timing_signal
        else:
            timing_signal = timing_signal.expand(x.size(0), -1, -1)
            x = torch.cat([x, timing_signal], -1)
        return self.dropout(x)

# From repo:
# https://github.com/pytorch/examples/blob/master/word_language_model/
class AttnEncoder(nn.Module):
    def __init__(self, ninp, nhead, nhid, pos_nhid, nlayers, max_len=512,
            pos_mode='absolute', pos_comb='cat', dropout=0.1, kvdim=64):
        super(AttnEncoder, self).__init__()
        self.src_mask = None
        self.kvdim = kvdim
        pos_inp = ninp if pos_comb == 'add' else pos_nhid
        self.pos_encoder = PositionalEncoding(pos_inp, mode=pos_mode, 
                comb=pos_comb, dropout=dropout, max_len=max_len)
        enc_inp = ninp if pos_comb == 'add' else ninp + pos_nhid
        self.pre_encoder = nn.Linear(enc_inp, kvdim)
        init_module_weights(self.pre_encoder, 0.1)
        encoder_layers = TransformerEncoderLayer(d_model=kvdim, nhead=nhead, 
                dim_feedforward=nhid, dropout=dropout)
        self.attn_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.pos_inp = pos_inp
        self.pos_nhid = pos_nhid
        self.enc_inp = enc_inp
        self.nhid = nhid
        self.nlayers = nlayers
        self.max_len = max_len

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, mask=None):
        src = self.pos_encoder(src)
        src = self.pre_encoder(src)
        # NOTE: need to reshape input to [seq_len, batch_size, dim] 
        output = self.attn_encoder(src.transpose(0, 1), mask)
        return output.transpose(0, 1)

# TODO
class SpeechTransformerLabeler(nn.Module):
    def __init__(self, config, tokenizer, label_tokenizer, 
            model_size="bert-base-uncased", 
            cache_dir="/s0/ttmt001", freeze='all'):
        super(SpeechTransformerLabeler, self).__init__()

        # Speech encoding attributes
        self.feature_types = config.feature_types
        self.feat_sizes = config.feat_sizes
        self.conv_sizes = config.conv_sizes
        self.num_conv = config.num_conv
        self.d_pause_embedding = config.d_pause_embedding
        self.pause_vocab = config.pause_vocab
        self.d_speech = config.d_speech if config.feature_types else 0
        self.fixed_word_length = config.fixed_word_length

        # Model attributes
        self.history_len = config.history_len
        self.num_labels = len(label_tokenizer)
        self.nhead = config.nhead
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.n_encoder_layers = config.n_encoder_layers
        self.pos_encoder_hidden_dim = config.pos_encoder_hidden_dim
        self.seq_max_len = config.seq_max_len
        self.dropout = config.dropout if hasattr(config, "dropout") else 0.0

        # Submodule configs
        self.pos_mode = config.pos_mode
        self.pos_comb = config.pos_comb
        self.hist_out = config.hist_out
        self.pooling_mode_cls_token = config.pooling_mode_cls_token
        self.pooling_mode_max_tokens = config.pooling_mode_max_tokens
        self.pooling_mode_mean_tokens = config.pooling_mode_mean_tokens
        self.attr_embedding_dim = config.attr_embedding_dim
        self.kvdim = config.kvdim

        # Vocabulary attributes
        self.word2id = tokenizer.word2id
        self.id2word = tokenizer.id2word
        self.label2id = label_tokenizer.word2id
        self.id2label = label_tokenizer.id2word
        self.vocab_size = len(tokenizer)
        self.label_vocab_size = len(label_tokenizer)
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_label_id = label_tokenizer.pad_token_id

        # Encoding components
        # Bert specific attributes
        self.word_embedding = BertEmbedder(model_size, cache_dir=cache_dir, 
                freeze=freeze)
        self.word_embedding_dim = self.word_embedding.embedding_size

        # Speech encoder attributes
        if self.feature_types:
            self.speech_encoder = SpeechFeatureEncoder(
                self.feature_types, self.feat_sizes,
                self.d_speech, 
                conv_sizes=self.conv_sizes,
                num_conv=self.num_conv,
                d_pause_embedding=self.d_pause_embedding,
                pause_vocab_len=len(self.pause_vocab),
                fixed_word_length=self.fixed_word_length,
                speech_dropout=self.dropout)
        else:
            self.speech_encoder = None

        # sentence encoder
        self.sent_encoder = AttnEncoder(
                self.word_embedding_dim + self.d_speech, 
                self.nhead, self.encoder_hidden_dim, 
                self.pos_encoder_hidden_dim, 
                self.n_encoder_layers, 
                max_len=self.seq_max_len,
                pos_mode=self.pos_mode, 
                pos_comb=self.pos_comb,
                dropout=self.dropout,
                kvdim=self.kvdim)

        # history encoder
        #self.history_encoder = Pooling(
        #        self.sent_encoder.enc_inp, 
        #        self.hist_out,
        #        pooling_mode_cls_token=self.pooling_mode_cls_token,
        #        pooling_mode_max_tokens=self.pooling_mode_max_tokens,
        #        pooling_mode_mean_tokens=self.pooling_mode_mean_tokens)
        
        if self.history_len > 1:
            self.history_encoder = Pooling(
                self.kvdim, 
                self.hist_out,
                pooling_mode_cls_token=self.pooling_mode_cls_token,
                pooling_mode_max_tokens=self.pooling_mode_max_tokens,
                pooling_mode_mean_tokens=self.pooling_mode_mean_tokens)
        else:
            self.history_encoder = None

        #dec_inp = self.sent_encoder.enc_inp + (self.history_len - 1) * self.hist_out
        dec_inp = self.kvdim + (self.history_len - 1) * self.hist_out
        self.dec_inp = dec_inp

        # floor encoder
        # NOTE: temp remove this
        #self.floor_encoder = RelSpeakerEncoder(
        #    input_dim=dec_inp,
        #    embedding_dim=self.attr_embedding_dim,
        #    hist_len=self.history_len)

        # Decoding components
        self.decoder = nn.Linear(dec_inp, self.num_labels)
        init_module_weights(self.decoder)


    def _get_attn_mask(self, attn_keys):
        attn_mask = (attn_keys != self.pad_token_id)
        return attn_mask

    def _encode(self, data):
        X_data = data["X"]
        seq_len = X_data.size(-1)
        batch_size = X_data.size(0) // self.history_len
        X = X_data.view(batch_size, self.history_len, -1)
        X_type_ids = data["X_type_ids"] 
        X_attn_masks = data["X_attn_masks"]
        input_floors = data["X_floor"]
        Xtext = self.word_embedding(X_data, X_type_ids, X_attn_masks)
        
        if self.speech_encoder is not None:
            Xspeech = self.speech_encoder(data["X_speech"])
            embedded_inputs = torch.cat([Xtext, Xspeech], -1)
        else:
            embedded_inputs = Xtext

        hist_len = self.history_len

        dim0 = batch_size * self.nhead
        mh_attn_masks = [(1 - X_attn_masks[i, :]).expand(seq_len, -1).unsqueeze(0).expand(self.nhead, -1, -1) for i in range(batch_size)]
        mh_attn_masks = torch.cat(mh_attn_masks, 0).bool()
        encoded = self.sent_encoder(embedded_inputs, mh_attn_masks)
        #encoded = self.sent_encoder(embedded_inputs)

        if self.history_encoder is not None:
            # Mask input to 0 so pooling only is done on history
            hist_mask = X_attn_masks.clone()
            hist_mask[hist_len-1::hist_len,:] = 0
            pooled_history = self.history_encoder(encoded, hist_mask)
            embedded_hist = pooled_history.view(batch_size, hist_len, 
                    pooled_history.size(-1))[:, :hist_len-1, :]
            embedded_hist = embedded_hist.view(batch_size, -1).unsqueeze(1).expand(-1, seq_len, -1)
            encoded_to_decode = encoded[hist_len-1::hist_len,:]
            word_encodings = torch.cat([encoded_to_decode, embedded_hist], -1)
        else:
            word_encodings = encoded
        
        input_lens = (X != self.pad_token_id).sum(-1)
        dial_lens = (input_lens > 0).long().sum(1)  # num non-padding sents

        # fetch target-sentence-releated information
        tgt_floors = []
        for dial_idx, dial_len in enumerate(dial_lens):
            tgt_floors.append(input_floors[dial_idx, dial_len-1])

        tgt_floors = torch.stack(tgt_floors, 0)
        src_floors = input_floors.view(-1)
        tgt_floors = tgt_floors.unsqueeze(1).repeat(1,self.history_len).view(-1)
        #word_encodings = self.floor_encoder(
        #    word_encodings,
        #    src_floors=src_floors,
        #    tgt_floors=tgt_floors
        #)
        
        return word_encodings, batch_size

    def _decode(self, word_encodings):
        batch_size = word_encodings.size(0)
        feats = word_encodings[:, 1:-1, :].contiguous()  # excluding bos, eos
        outputs = self.decoder(feats)
        ret_dict = {}
        ret_dict["logits"] = outputs
        ret_dict["symbols"] = outputs.topk(1)[1]
        return ret_dict

    def train_step(self, data):
        # Forward
        word_encodings, batch_size = self._encode(data)
        Y = data["Y"]
        Y_out = Y[:, 1:-1].contiguous() # don't count eos in loss
        max_y_len = Y_out.size(1)

        decoder_ret_dict = self._decode(word_encodings)

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
            "loss": loss,
            "symbols": decoder_ret_dict["symbols"]
        }
        ret_stat = {
            "loss": loss.item()
        }

        return ret_data, ret_stat

    def evaluate_step(self, data):
        with torch.no_grad():
            # Forward
            word_encodings, batch_size = self._encode(data)
            Y = data["Y"]
            Y_out = Y[:, 1:-1].contiguous()
            max_y_len = Y_out.size(1)
            decoder_ret_dict = self._decode(word_encodings)

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
        ret_data = {"symbols": decoder_ret_dict["symbols"]}
        ret_stat = {
            "monitor": loss.item(),
            "loss": loss.item()
        }

        return ret_data, ret_stat

    def test_step(self, data):
        loss = None
        with torch.no_grad():
            # Forward
            word_encodings, batch_size = self._encode(data)
            decoder_ret_dict = self._decode(word_encodings)
            if "Y" in data:
                Y = data["Y"]
                Y_out = Y[:, 1:-1].contiguous()
                max_y_len = Y_out.size(1)
                logits = decoder_ret_dict["logits"]
                label_losses = F.cross_entropy(
                    logits.view(-1, self.label_vocab_size),
                    Y_out.view(-1), ignore_index=self.pad_label_id,
                    reduction="none"
                ).view(batch_size, max_y_len)
                loss = label_losses.sum()
                loss = loss.item()

        ret_data = {
            "symbols": decoder_ret_dict["symbols"],
            "batch_loss": loss
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
        self.d_speech = config.d_speech if config.feature_types else 0
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
        if self.feature_types:
            self.speech_encoder = SpeechFeatureEncoder(
                self.feature_types, self.feat_sizes,
                self.d_speech, 
                conv_sizes=self.conv_sizes,
                num_conv=self.num_conv,
                d_pause_embedding=self.d_pause_embedding,
                pause_vocab_len=len(self.pause_vocab),
                fixed_word_length=self.fixed_word_length,
                speech_dropout=self.dropout)
        else:
            self.speech_encoder = None

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
        
        if self.speech_encoder is not None:
            Xspeech = self.speech_encoder(data["X_speech"])
            embedded_inputs = torch.cat([Xtext, Xspeech], -1)
        else:
            embedded_inputs = Xtext

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

