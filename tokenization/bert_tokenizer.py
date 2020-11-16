import code
import collections

import torch
from transformers import BertTokenizer


class ModBertTokenizer(object):
    def __init__(self, model_size, pad_token="[PAD]",
                 unk_token="[UNK]", sep_token="[SEP]",
                 cls_token="[CLS]", mask_token="[MASK]", 
                 cache_dir="/s0/ttmt001"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.mask_token = mask_token

        # load from pretrained tokenizer
        assert model_size in ["base", "large"]
        self.pretrained = BertTokenizer.from_pretrained(f"bert-{model_size}-uncased", 
                cache_dir=cache_dir)

        # vocab dict and revserse vocab dict
        self.word2id = self.pretrained.vocab
        self.id2word = self.pretrained.ids_to_tokens

        # set special token ids
        for token_type in ["pad_token", "unk_token", "sep_token", "cls_token", "mask_token"]:
            token = getattr(self, token_type)
            setattr(self, f"{token_type}_id", self.word2id[token])

    def __len__(self):
        return len(self.word2id)

    def convert_tokens_to_string(self, tokens):
        sent = self.pretrained.decode(self.pretrained.convert_tokens_to_ids(tokens))
        return sent

    def convert_string_to_tokens(self, sent):
        if len(sent) == 0:
            return []
        else:
            return self.pretrained.tokenize(sent)

    def convert_tokens_to_ids(self, tokens):
        ids = self.pretrained.convert_tokens_to_ids(tokens)
        return ids

    def convert_ids_to_tokens(self, ids, trim_pad=False, **kwargs):
        _tokens = self.pretrained.convert_ids_to_tokens(ids)
        tokens = []
        for token in _tokens:
            if trim_pad and token == self.id2word[self.pad_token_id]:
                continue
            tokens.append(token)
        return tokens

    def convert_batch_ids_to_tensor(self, batch_ids, history_len):
        """Turning a list token id sequences `batch_ids` into a mini-batch tensor.
        Sequences are right-padded with `self.pad_token_id`.
        """
        batch_lens = [len(ids) for ids in batch_ids]
        max_len = max(batch_lens)
        batch_size = len(batch_ids)//history_len

        padded_batch_ids = [ids + [self.pad_token_id]*(max_len-len(ids)) for ids in batch_ids]
        batch_tensor = torch.LongTensor(padded_batch_ids)
        
        type_ids = []
        for _ in range(batch_size):
            type_ids += [[0]*max_len]*(history_len - 1) + [[1]*max_len]
        batch_type_ids = torch.LongTensor(type_ids)

        attn_masks = [[1]*len(ids) + [0]*(max_len-len(ids)) for ids in batch_ids]
        batch_attn_masks = torch.LongTensor(attn_masks)

        return batch_tensor, batch_type_ids, batch_attn_masks
