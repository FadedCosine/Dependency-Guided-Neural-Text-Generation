import os
import torch
import numpy as np
from torch.utils.data import Dataset
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
        self.dictionary.add_word('<pad>')

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        lengths = []
        with open(path, 'r') as f:
            for line in f:
                # 加bos是基于sentence训练的尝试，但是失败了
                # words = ['<bos>'] + line.split() + ['<eos>']
                words = line.split() + ['<eos>']
                lengths.append(len(words))
                for word in words:
                    self.dictionary.add_word(word)
        lengths = torch.LongTensor(lengths)
        # Tokenize file content
        ids = []
        with open(path, 'r') as f:
            for line in f:
                cur_line_ids = []
                # 加bos是基于sentence训练的尝试，但是失败了
                # words = ['<bos>'] + line.split() + ['<eos>']
                words = line.split() + ['<eos>']
                for word in words:
                    cur_line_ids.append(self.dictionary.word2idx[word])
                ids.append(cur_line_ids)

        return np.array(ids), lengths

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
        if len(self.train_dependency_head) == 0:
            raise FileNotFoundError(f"Dataset dependency not found: {dependency_path}")
        
        self.train_dep_token_list = self.build_dependency_token_list(self.train, self.train_lengths, self.train_dependency_head)
        self.valid_dep_token_list = self.build_dependency_token_list(self.valid, self.valid_lengths, self.valid_dependency_head)
        self.test_dep_token_list = self.build_dependency_token_list(self.test, self.test_lengths, self.test_dependency_head)
        # self.train_dep_indicators_allinone = self.build_dependency_indicators(self.train, self.train_lengths, self.train_dependency_head)
        # self.valid_dep_indicators_allinone = self.build_dependency_indicators(self.valid, self.valid_lengths, self.valid_dependency_head)
        # self.test_dep_indicators_allinone = self.build_dependency_indicators(self.test, self.test_lengths, self.test_dependency_head)
        # dependency_token_list: [  [ [cur_position_dependency_token1, cur_position_dependency_token2, ... ] * seqlen1  ] * lines ]
        self.train_allinone = [self.dictionary.word2idx['<eos>']]
        for sent_list in self.train:
            self.train_allinone.extend(sent_list)
        self.train_allinone = self.train_allinone[:-1] # 在首部加上一个eos，删除尾部的eos，这一步使得train_allinone和train_dep_token_allinone直接对齐
        self.valid_allinone = [self.dictionary.word2idx['<eos>']]
        for sent_list in self.valid:
            self.valid_allinone.extend(sent_list)
        self.valid_allinone = self.valid_allinone[:-1]
        self.test_allinone = [self.dictionary.word2idx['<eos>']]
        for sent_list in self.test:
            self.test_allinone.extend(sent_list)
        self.test_allinone = self.test_allinone[:-1]

        # self.train_dep_tokens_lists = []
        # for dep_tokens_list in self.train_dep_token_list:
        #     self.train_dep_tokens_lists.extend(dep_tokens_list)
        # self.valid_dep_tokens_lists = []
        # for dep_tokens_list in self.valid_dep_token_list:
        #     self.valid_dep_tokens_lists.extend(dep_tokens_list)
        # self.test_dep_tokens_lists = []
        # for dep_tokens_list in self.test_dep_token_list:
        #     self.test_dep_tokens_lists.extend(dep_tokens_list)
        
        assert len(self.train_allinone) == len(self.train_dep_token_list)
        assert len(self.valid_allinone) == len(self.valid_dep_token_list)
        assert len(self.test_allinone) == len(self.test_dep_token_list)
        self.train_allinone = torch.LongTensor(self.train_allinone)
        self.valid_allinone = torch.LongTensor(self.valid_allinone)
        self.test_allinone = torch.LongTensor(self.test_allinone)
        # self.train_dep_indicators_allinone = torch.LongTensor(self.train_dep_indicators_allinone)
        # self.valid_dep_indicators_allinone = torch.LongTensor(self.valid_dep_indicators_allinone)
        # self.test_dep_indicators_allinone = torch.LongTensor(self.test_dep_indicators_allinone)
        
    def build_dependency_indicators(self, token_list, len_list, head_list):
        total_len = sum(len_list)
        dependency_indicators = torch.zeros((total_len, len(self.dictionary)))
        gloab_idx = 0
        for idx in range(len(token_list)):
            cur_dep_token_list = [[] for _ in range(len_list[idx])]
            source = [self.dictionary.word2idx['<eos>']] + token_list[idx][:-1]
            cur_dep_token_list[-1].append(self.dictionary.word2idx['<eos>'])
            # print(" len of source is : ", len(source))
            # print(" len of head_list[idx] is : ", len(head_list[idx]))
            for i, head in enumerate(head_list[idx]):
                cur_idx = i + 1
                if cur_idx < head:
                    cur_dep_token_list[cur_idx].append(source[head])
                elif cur_idx > head:
                    cur_dep_token_list[head].append(source[cur_idx])
                else:
                    raise ValueError("Improssible! One token's dependency head is itself.")
            for line_idx, dep_set in enumerate(dep_tokens_list):
                dependency_indicators[gloab_idx+line_idx, dep_set] = 1
            gloab_idx += len_list[idx]  
        return dependency_indicators
    def build_dependency_token_list(self, token_list, len_list, head_list):
        """
        构造self.train等的dependency token list
        """
        dependency_token_list = []
        for idx in range(len(token_list)):
            cur_dep_token_list = [set() for _ in range(len_list[idx])]
            source = [self.dictionary.word2idx['<eos>']] + token_list[idx][:-1]
         
            cur_dep_token_list[-1].add(self.dictionary.word2idx['<eos>'])
            # print(" len of source is : ", len(source))
            # print(" len of head_list[idx] is : ", len(head_list[idx]))
            for i, head in enumerate(head_list[idx]):
                cur_idx = i + 1
                if cur_idx < head:
                    cur_dep_token_list[cur_idx].add(source[head])
                elif cur_idx > head:
                    cur_dep_token_list[head].add(source[cur_idx])
                else:
                    raise ValueError("Improssible! One token's dependency head is itself.")
            dependency_token_list.extend(cur_dep_token_list)
            #dependency_token_list只是每个句子的每个位置的于其有dependency关系的token id 的list，之后在batchify中转换成dependency_set_indicator
        return np.array(dependency_token_list)

"""
尝试过上述sentence级别的训练之后发现，LSTM不能做到像Transformer LM那样Unconditional Text Generation，loss 根本降不下来。
因此可以在使用head文件得到dependency token list之后，整合到一个list中，依然进行原来那样language modeling的训练
"""

class SentenceDataset(Dataset):
    def __init__(self, sentence_data, data_lengths, dictionary):
        self.sorted_lengths, self.sorted_idx = data_lengths.sort(0, descending=True)
        self.sentence_data = sentence_data[self.sorted_idx]
        self.dictionary = dictionary
    def __len__(self):
        return len(self.sentence_data)
    def __getitem__(self, index):
        source = torch.LongTensor(self.sentence_data[index][:-1])
        target = torch.LongTensor(self.sentence_data[index][1:])
        
        return {"id": index, "source": source, "target": target}

class SentenceWithDependencyDataset(Dataset):
    def __init__(self, sentence_data, data_lengths, dependency_token_data, dictionary):
        self.sorted_lengths, self.sorted_idx = data_lengths.sort(0, descending=True)
        self.sentence_data = sentence_data[self.sorted_idx]
        self.dependency_token_data = dependency_token_data[self.sorted_idx]
        self.dictionary = dictionary
    def __len__(self):
        return len(self.sentence_data)
    def __getitem__(self, index):
        source = torch.LongTensor(self.sentence_data[index][:-1])
        target_set_list = self.dependency_token_data[index]
        seq_len = source.size()[-1]
        target = torch.zeros((seq_len, self.dictionary.length()))
        
        for idx, target_set in enumerate(target_set_list):
            target[idx, target_set] = 1
        return {"id": index, "source": source, "target": target}