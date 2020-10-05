# Originally written by the AWD-LSTM team
# Rewritten and modified heavily by Luc Hayward (HYWLUC001) to allow for flexible BPE libraries provided by HuggingFace

import os
import torch

from collections import Counter
from tokenizers import ByteLevelBPETokenizer


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0
        self.avg_characters_per_token = {'train': -1, 'valid': -1, 'test': -1}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

# -------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------Written by Luc Hayward------------------------------------------------------ #
class Corpus(object):
    def __init__(self, path, vocab_size=-1, use_bpe=False, tokenizer_data=""):
        self.dictionary = Dictionary()

        if use_bpe:
            assert os.path.exists(path), "Path does not exist: " + path

            print("-------------------------------------------------------------")

            tokenizer = ByteLevelBPETokenizer()
            if len(tokenizer_data) != 0:
                print("Training tokenizer on: "+os.path.join(tokenizer_data, 'train.txt'))
                tokenizer.train(
                    [os.path.join(tokenizer_data, 'train.txt')],
                    vocab_size=vocab_size,
                    show_progress=False
                )
            else:
                print("Training tokenizer on: "+os.path.join(path, 'train.txt'))
                tokenizer.train(
                    [
                        os.path.join(path, 'train.txt')
                        # os.path.join(path, 'valid.txt'),
                        # os.path.join(path, 'test.txt')
                    ],
                    vocab_size=vocab_size,
                    show_progress=False
                )
            print("-------------------------------------------------------------")

            print("Encoding dataset at: " + path)
            with open(os.path.join(path, 'train.txt'), 'r', encoding='utf-8') as f:
                text = f.read()
                enc = tokenizer.encode(text)
                tokens = len(enc.ids)
                ids = torch.LongTensor(tokens)

                for index, id in enumerate(enc.ids):
                    ids[index] = id
                self.train = ids
                self.dictionary.avg_characters_per_token['train'] = len(text) / len(enc.ids)

            with open(os.path.join(path, 'valid.txt'), 'r', encoding='utf-8') as f:
                text = f.read()
                enc = tokenizer.encode(text)
                tokens = len(enc.ids)
                ids = torch.LongTensor(tokens)

                for index, id in enumerate(enc.ids):
                    ids[index] = id
                self.valid = ids
                self.dictionary.avg_characters_per_token['valid'] = len(text) / len(enc.ids)

            with open(os.path.join(path, 'test.txt'), 'r', encoding='utf-8') as f:
                text = f.read()
                enc = tokenizer.encode(text)
                tokens = len(enc.ids)
                ids = torch.LongTensor(tokens)

                for index, id in enumerate(enc.ids):
                    ids[index] = id
                self.test = ids
                self.dictionary.avg_characters_per_token['test'] = len(text) / len(enc.ids)
            print("-------------------------------------------------------------")

            self.dictionary.word2idx = tokenizer.get_vocab()
            self.dictionary.idx2word = [tokenizer.id_to_token(x) for x in range(tokenizer.get_vocab_size())]
            self.dictionary.total = tokenizer.get_vocab_size()


        else:
            self.train = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize(os.path.join(path, 'test.txt'))

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path), "Path does not exist: " + path
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
