import numpy as np
import torch
from fairseq.data import FairseqDataset, plasma_utils, data_utils
from fairseq.data.indexed_dataset import best_fitting_int_dtype
from typing import Tuple

class SentenceWithDependencyDataset(FairseqDataset):
    """Build a dataset with sentences and its corresponding dependency head lists;
    use for dependency decoder, to predict dependent tokens with context
    """
    def __init__(
        self,
        dataset,
        sizes,
        dependency_token_lists,
        pad,
        eos,
        document_sep_len=1,
        use_plasma_view=False,
        split_path=None,
        plasma_path=None,
        break_mode='eos',
        block_size=0,
    ):
        super().__init__()
        """
        this code based on fairseq.data.token_block_dataset 's TokenBlockDataset,
        but for our task, we only use the sentence with dependency as our data item.
        So the document_sep_len is set to 1, if break_mode is 'eos', block_size is set to 0
        """
        self.break_mode = break_mode
        self.document_sep_len = document_sep_len
        if break_mode == 'eos':
            self.block_size = 0
        else:
            self.block_size = block_size
        """
        这里self.dataset[idx]是txt中第idx+1行的token_list；
        sizes[idx]即为txt中第idx+1行的token_list的len
        """
        self.dataset = dataset
        # self.dataset_dependency = dependency_token_lists
        self.dependency_token_lists = dependency_token_lists
        
        self.pad = pad
        self.eos = eos
        assert len(dataset) > 0
        assert len(dataset) == len(sizes)
        _sizes, block_to_dataset_index, slice_indices = self._build_slice_indices(
            sizes, self.break_mode, self.document_sep_len, self.block_size
        )
        if use_plasma_view:
            plasma_id = (self.block_size, self.document_sep_len, str(self.break_mode), len(dataset))
            self._slice_indices = plasma_utils.PlasmaView(
                slice_indices, split_path, (plasma_id, 0), plasma_path=plasma_path
            )
            self._sizes = plasma_utils.PlasmaView(
                _sizes, split_path, (plasma_id, 1), plasma_path=plasma_path
            )
            self._block_to_dataset_index = plasma_utils.PlasmaView(
                block_to_dataset_index, split_path, (plasma_id, 2), plasma_path=plasma_path,
            )
        else:
            self._slice_indices = plasma_utils.PlasmaArray(slice_indices)
            self._sizes = plasma_utils.PlasmaArray(_sizes)
            self._block_to_dataset_index = plasma_utils.PlasmaArray(
                block_to_dataset_index
            )
        
        # self.dependency_token_lists = self._build_dependency_token_list(dataset, dataset_dependency, sizes)
    
    @staticmethod
    def _build_slice_indices(
        sizes, break_mode, document_sep_len, block_size
    ) -> Tuple[np.ndarray]:
        """Use token_block_utils_fast to build arrays for indexing into self.dataset"""
        try:
            from fairseq.data.token_block_utils_fast import (
                _get_slice_indices_fast,
                _get_block_to_dataset_index_fast,
            )
        except ImportError:
            raise ImportError(
                "Please build Cython components with: `pip install --editable .` "
                "or `python setup.py build_ext --inplace`"
            )

        if isinstance(sizes, list):
            sizes = np.array(sizes, dtype=np.int64)
        else:
            if torch.is_tensor(sizes):
                sizes = sizes.numpy()
            sizes = sizes.astype(np.int64)

        break_mode = break_mode if break_mode is not None else "none"

        # For "eos" break-mode, block_size is not required parameters.
        if break_mode == "eos" and block_size is None:
            block_size = 0

        slice_indices = _get_slice_indices_fast(
            sizes, str(break_mode), block_size, document_sep_len
        )
        _sizes = slice_indices[:, 1] - slice_indices[:, 0]

        # build index mapping block indices to the underlying dataset indices
        if break_mode == "eos":
            # much faster version for eos break mode
            block_to_dataset_index = np.stack(
                [
                    np.arange(len(sizes)),  # starting index in dataset
                    np.zeros(
                        len(sizes), dtype=np.compat.long
                    ),  # starting offset within starting index
                    np.arange(len(sizes)),  # ending index in dataset
                ],
                1,
            )
        else:
            block_to_dataset_index = _get_block_to_dataset_index_fast(
                sizes, slice_indices,
            )
        size_dtype = np.uint16 if block_size < 65535 else np.uint32
        num_tokens = slice_indices[-1].max()
        slice_indices_dtype = best_fitting_int_dtype(num_tokens)
        slice_indices = slice_indices.astype(slice_indices_dtype)
        """
        如果把datasets中所有句子的tokens并到同一个list中，slice_indices对应的就是合并前每个句子在这个list中的range
        
        """
        _sizes = _sizes.astype(size_dtype)
        block_to_dataset_index = block_to_dataset_index.astype(slice_indices_dtype)
        """
        对于eos模式来说，这个block_to_dataset_index的就是[[i,0,i]]
        """
        return _sizes, block_to_dataset_index, slice_indices
    @property
    def slice_indices(self):
        return self._slice_indices.array

    @property
    def sizes(self):
        return self._sizes.array

    @property
    def block_to_dataset_index(self):
        return self._block_to_dataset_index.array
    
    def attr(self, attr: str, index: int):
        start_ds_idx, _, _ = self.block_to_dataset_index[index]
        return self.dataset.attr(attr, start_ds_idx)
    
    def __len__(self):
        return len(self.slice_indices)
    # def __getitem__(self, index):
    #     start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]

    #     buffer = torch.cat(
    #         [self.dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)]
    #     )
    #     slice_s, slice_e = self.slice_indices[index]
    #     length = slice_e - slice_s
    #     s, e = start_offset, start_offset + length
    #     item = buffer[s:e]
    #     item_dependency = self.dataset_dependency[index]
    #     """
    #     每个item相较于原句要在尾部多一个eos，这是在fairseq-preprocess的时候加上的;
    #     另外，我期望self.dataset_dependency[index]依然是 和syndehead文件中的每一行一模一样的列表
    #     """

    #     source = torch.cat([item.new([self.eos]), buffer[0 : e - 1]])
    #     dependency_token_list = [[] for _ in range(length)]
    #     dependency_token_list[-1].append(self.eos)
    #     for i, head in enumerate(item_dependency):
    #         cur_idx = i + 1
    #         if cur_idx < head:
    #             dependency_token_list[cur_idx].append(source[head].item())
    #         elif cur_idx > head:
    #             dependency_token_list[head].append(source[cur_idx].item())
    #         else:
    #             raise ValueError("Improssible! One token's dependency head is itself.")
    #     #这里没用vocab，所以在下一个WrapDependencyDataset中把dependency_token_list转换成dependency_set_indicator
    #     return source, dependency_token_list
    def __getitem__(self, index):
        #! 这里存在一些bug，使得多卡训不起来
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]

        buffer = torch.cat(
            [self.dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)]
        )
        slice_s, slice_e = self.slice_indices[index]
        length = slice_e - slice_s
        s, e = start_offset, start_offset + length
        item = buffer[s:e]
        """
        每个item相较于原句要在尾部多一个eos，这是在fairseq-preprocess的时候加上的;
        另外，我期望self.dataset_dependency[index]依然是 和syndehead文件中的每一行一模一样的列表
        """
        # 这里在首部加eos构造source是fairseq的历史遗留问题
        if s == 0:
            source = torch.cat([item.new([self.eos]), buffer[s : e - 1]])
        else:
            source = buffer[s - 1 : e - 1]
      
        dependency_buffer = [dep_item for idx in range(start_ds_idx, end_ds_idx + 1) for dep_item in self.dependency_token_lists[idx] ]
 
        dependency_token_item = dependency_buffer[s:e]
       
        if self.break_mode == "eos":
            assert dependency_token_item == self.dependency_token_lists[index]
        assert source.size(-1) == len(dependency_token_item)
        return source, dependency_token_item
    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        self.dataset.prefetch(
            {
                ds_idx
                for index in indices
                for start_ds_idx, _, end_ds_idx in [self.block_to_dataset_index[index]]
                for ds_idx in range(start_ds_idx, end_ds_idx + 1)
            }
        )
            
def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}
    def merge(key):
        if key == "source":
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad=False,
            )
        elif key == "target":
            values = [s[key] for s in samples]
            vocab_size = values[0].size(-1)
            size = max(v.size(0) for v in values)
            for i in range(len(values)):
                values[i] = torch.cat([values[i], torch.zeros(size - values[i].size(0), vocab_size)], dim=0)
            return torch.stack(values)

    #src_tokens即为batch中的sources，依据这个batch最长的sources经过padding之后得到的2d tensor
    src_tokens = merge("source")
  
    #此时会是samples[0]["target"]是tensor,[seq_len, vocab_size]
    target = merge("target")
    #返回的target是[batch_size, seq_batch_max_len, vocab_size]
    return {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "nsentences": len(samples),
        "ntokens": sum(len(s["source"]) for s in samples),
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": torch.LongTensor([s["source"].numel() for s in samples]),
        },
        "target": target,
        "seq_true_lengths": torch.LongTensor([s["source"].numel() for s in samples]),
    }


class WrapDependencyDataset(FairseqDataset):
    """
    A wrapper around torch.utils.data.Dataset for DependencyDataset, used for min-batch
    """
    def __init__(
        self,
        dataset,
        sizes,
        src_vocab,
        tgt_vocab=None,
        add_eos_for_other_targets=False,
        shuffle=False,
    ):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = src_vocab
        self.tgt_vocab = tgt_vocab or src_vocab
        self.add_eos_for_other_targets = add_eos_for_other_targets
        self.shuffle = shuffle
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        source, target_set_list = self.dataset[index]
        seq_len = source.size()[-1]
        target = torch.zeros((seq_len, len(self.vocab)))
        
        for idx, target_set in enumerate(target_set_list):
            target[idx, target_set] = 1
        return {"id": index, "source": source, "target": target}

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the right.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the right.
        """
        return collate(samples, self.vocab.pad(), self.vocab.eos())
        
    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        self.dataset.prefetch(indices)

