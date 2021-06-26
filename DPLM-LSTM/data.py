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

        return np.array(ids), lengths

class DependencyCorpus(CorpusSentence):
    def __init__(self, data_path, dependency_path, train_filename, vaild_filename, test_filename):
        super().__init__(data_path)
        train_dataset_dependency = []
        with open(os.path.join(dependency_path, train_filename), "r", encoding="utf-8") as f:
            for line in f:
                train_dataset_dependency.append(eval(line.strip("\n")))
        valid_dataset_dependency = []
        with open(os.path.join(dependency_path, vaild_filename), "r", encoding="utf-8") as f:
            for line in f:
                valid_dataset_dependency.append(eval(line.strip("\n")))
        test_dataset_dependency = []
        with open(os.path.join(dependency_path, test_filename), "r", encoding="utf-8") as f:
            for line in f:
                test_dataset_dependency.append(eval(line.strip("\n")))
        if len(train_dataset_dependency) == 0:
            raise FileNotFoundError(f"Dataset dependency not found: {dependency_path}")
    def build_dependency_token_list():
        """
        构造self.train等的dependency token list
        """
        # 这里在首部加eos构造source是fairseq的历史遗留问题
        source = torch.cat([item.new([self.eos]), buffer[0 : e - 1]])
        dependency_token_list = [[] for _ in range(length)]
        dependency_token_list[-1].append(self.eos)
        for i, head in enumerate(item_dependency):
            cur_idx = i + 1
            if cur_idx < head:
                dependency_token_list[cur_idx].append(source[head].item())
            elif cur_idx > head:
                dependency_token_list[head].append(source[cur_idx].item())
            else:
                raise ValueError("Improssible! One token's dependency head is itself.")
        #这里没用vocab，所以在下一个WrapDependencyDataset中把dependency_token_list转换成dependency_set_indicator

