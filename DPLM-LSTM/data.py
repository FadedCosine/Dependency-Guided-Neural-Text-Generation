import os
import torch
import numpy as np
from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

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


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class CorpusSentence(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train, self.train_lengths = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid, self.valid_lengths  = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test, self.test_lengths  = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        lengths = []
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                lengths.append(len(words))
                for word in words:
                    self.dictionary.add_word(word)
        lengths = lengths.torch.LongTensor(lengths)
        # Tokenize file content
        ids = []
        with open(path, 'r') as f:
            token = 0
            for line in f:
                cur_line_ids = []
                words = line.split() + ['<eos>']
                for word in words:
                    cur_line_ids.append(self.dictionary.word2idx[word])
                ids.append(cur_line_ids)

        return ids, lengths

class DependencyCorpus(CorpusSentence):
    def __init__(self, data_path, dependency_path, train_filename, vaild_filename, test_filename):
        super().__init__(data_path)
        self.train_dependency_head = []
        with open(os.path.join(dependency_path, train_filename), "r", encoding="utf-8") as f:
            for line in f:
                self.train_dependency_head.append(eval(line.strip("\n")))
        self.valid_dependency_head = []
        with open(os.path.join(dependency_path, vaild_filename), "r", encoding="utf-8") as f:
            for line in f:
                self.valid_dependency_head.append(eval(line.strip("\n")))
        self.test_dependency_head = []
        with open(os.path.join(dependency_path, test_filename), "r", encoding="utf-8") as f:
            for line in f:
                self.test_dependency_head.append(eval(line.strip("\n")))
        if len(self.train_dataset_dependency) == 0:
            raise FileNotFoundError(f"Dataset dependency not found: {dependency_path}")
        
        self.train_dep_token_list = self.build_dependency_token_list(self.train, self.train_lengths, self.train_dependency_head)
        self.valid_dep_token_list = self.build_dependency_token_list(self.valid, self.valid_lengths, self.valid_dependency_head)
        self.test_dep_token_list = self.build_dependency_token_list(self.test, self.test_lengths, self.test_dependency_head)


    def build_dependency_token_list(token_list, len_list, head_list):
        """
        构造self.train等的dependency token list
        """
        dependency_token_list = []
        for idx in range(len(token_list)):
            cur_dep_token_list = [[] for _ in range(len_list[idx])]
            # 这里在首部加eos构造source是fairseq的历史遗留问题
            source = [self.dictionary.word2idx('<eos>')] + token_list[idx][:-1]
            cur_dep_token_list[-1].append(self.dictionary.word2idx('<eos>'))
            for i, head in enumerate(head_list[idx]):
                cur_idx = i + 1
                if cur_idx < head:
                    cur_dep_token_list[cur_idx].append(source[head].item())
                elif cur_idx > head:
                    cur_dep_token_list[head].append(source[cur_idx].item())
                else:
                    raise ValueError("Improssible! One token's dependency head is itself.")
            dependency_token_list.append(cur_dep_token_list)
            #dependency_token_list只是每个句子的每个位置的于其有dependency关系的token id 的list，之后在batchify中转换成dependency_set_indicator
    return cur_dep_token_list
