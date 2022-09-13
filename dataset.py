import os
import random
from torch.utils.data import Dataset
from utils import tokens_to_seq, contains_digit
from operator import itemgetter
from spacy.lang.en import English


class Language(object):
    def __init__(self, vocab_limit, data_pairs):
        self.data_pairs = data_pairs
        self.vocab = self.create_vocab()

        truncated_vocab = sorted(self.vocab.items(), key=itemgetter(1), reverse=True)[
            :vocab_limit]

        self.tok_to_idx = dict()
        self.tok_to_idx['<MSK>'] = 0
        self.tok_to_idx['<SOS>'] = 1
        self.tok_to_idx['<EOS>'] = 2
        self.tok_to_idx['<UNK>'] = 3
        for idx, (tok, _) in enumerate(truncated_vocab):
            self.tok_to_idx[tok] = idx + 4
        self.idx_to_tok = {idx: tok for tok, idx in self.tok_to_idx.items()}

    def create_vocab(self):
        vocab = dict()
        for dp in self.data_pairs:
            assert len(dp) == 2
            tokens = dp[0].split() + dp[1].split()
            for token in tokens:
                vocab[token] = vocab.get(token, 0) + 1
        return vocab


class SequencePairDataset(Dataset):
    def __init__(self,
                 data_pairs,
                 maxlen=64,
                 lang=None,
                 vocab_limit=None,
                 seed=42,
                 is_val=False,
                 use_cuda=False,
                 use_extended_vocab=True):
        """
        data_pairs: list of tuples (input_string, output_string)
        """

        self.data_pairs = data_pairs
        self.maxlen = maxlen
        self.use_cuda = use_cuda
        self.seed = seed
        self.is_val = is_val
        self.use_extended_vocab = use_extended_vocab

        if lang is None:
            lang = Language(vocab_limit, data_pairs)

        self.lang = lang

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        """
        :arg
        idx: int

        :returns
        input_token_list: list[int]
        output_token_list: list[int]
        token_mapping: binary array"""

        input_token_list = self.data_pairs[idx][0].split()
        output_token_list = self.data_pairs[idx][1].split()

        input_token_list = (['<SOS>'] + input_token_list +
                            ['<EOS>'])[:self.maxlen]
        output_token_list = (
            ['<SOS>'] + output_token_list + ['<EOS>'])[:self.maxlen]

        input_seq = tokens_to_seq(
            input_token_list, self.lang.tok_to_idx, self.maxlen, self.use_extended_vocab)
        output_seq = tokens_to_seq(output_token_list, self.lang.tok_to_idx,
                                   self.maxlen, self.use_extended_vocab, input_tokens=input_token_list)

        if self.use_cuda:
            input_seq = input_seq.cuda()
            output_seq = output_seq.cuda()

        return input_seq, output_seq, ' '.join(input_token_list), ' '.join(output_token_list)
