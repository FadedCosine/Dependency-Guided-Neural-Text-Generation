import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils


logger = logging.getLogger(__name__)
class LanguagePairWithDependencyDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets, the target side sequence has its corresponding dependency head lists
    """
    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        tgt_dependency_head=None,
        left_pad_source=True,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        if tgt is not None:
            assert len(src) == len(
                tgt
            ), "Source and target must contain the same number of examples"
        self.src = src
        self.tgt = tgt
        self.tgt_dependency_head = tgt_dependency_head
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        self.buckets = None
        self.pad_to_multiple = pad_to_multiple
    def get_batch_shapes(self):
        return self.buckets
    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        tgt_item_dependency = self.tgt_dependency_head[index] if self.tgt is not None else None
        src_item = self.src[index]
        
        """
        每个item相较于原句要在尾部多一个eos，这是在fairseq-preprocess的时候加上的;
        """
        if self.append_eos_to_target: #In seq2seq2 dependency modeling, always False
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos: #In seq2seq2 dependency modeling, always False
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.tgt and self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

            bos = self.src_dict.bos()
            if self.src[index][0] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])

        if self.remove_eos_from_source: #In seq2seq2 dependency modeling, always False
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]
        tgt_dependency_token_list = [[] for _ in range(self.tgt_sizes[index])]
        """
        length of tgt_item_dependency maybe  == self.tgt_sizes[index]-1;
        tgt_item_dependency[-1] is [eos]
        """
        for i, head in enumerate(tgt_item_dependency):
            cur_idx = i + 1
            if cur_idx < head:
                tgt_dependency_token_list[cur_idx].append(tgt_item[head-1].item())
            elif cur_idx > head:
                tgt_dependency_token_list[head].append(tgt_item[cur_idx-1].item())
            else:
                raise ValueError("Improssible! One token's dependency head is itself.")
        tgt_dependency_token_list[-1].append(self.tgt_dict.eos())

        tgt_dependency_indicator = torch.zeros((self.tgt_sizes[index], len(self.tgt_dict)))
        for idx, target_set in enumerate(tgt_dependency_token_list):
            target[idx, target_set] = 1
        example = {
            "id": index,
            "source": src_item,
            "target": tgt_dependency_indicator,
        }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example
    def __len__(self):
        return len(self.src)
    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.