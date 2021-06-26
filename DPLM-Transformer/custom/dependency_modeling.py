import numpy as np
import torch
import logging
from fairseq import utils
from dataclasses import dataclass, field
from typing import Optional
from fairseq.tasks import LegacyFairseqTask, register_task, FairseqTask
from fairseq.data import (
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    LMContextWindowDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    TruncatedDictionary,
    data_utils,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.data.sentence_with_dependency_dataset import SentenceWithDependencyDataset, WrapDependencyDataset
import os
from omegaconf import II


SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])
logger = logging.getLogger(__name__)

@dataclass
class DependencyModelingConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    output_dictionary_size: int = field(
        default=-1, metadata={"help": "limit the size of output dictionary"}
    )
    self_target: bool = field(default=False, metadata={"help": "include self target"})
    future_target: bool = field(
        default=False, metadata={"help": "include future target"}
    )
    past_target: bool = field(default=False, metadata={"help": "include past target"})
    add_bos_token: bool = field(
        default=False, metadata={"help": "prepend beginning of sentence token (<s>)"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    # TODO common vars below add to parent
    seed: int = II("common.seed")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")
    use_plasma_view: bool = II("common.use_plasma_view")
    plasma_path: str = II("common.plasma_path")

    
    dependency: str = field(
        default="/home/yangzhixian/DependencyGuided/DG/data/news/dependency",
        metadata={
            "help": "path to data dependency"
        },
    )
    dependency_suffix: str = field(
        default=".head",
        metadata={
            "help": "file suffix of dependency head file"
        },
    )

@register_task("dependency_modeling", dataclass=DependencyModelingConfig)
class DependencyModelingTask(LegacyFairseqTask):
    """
    Train a DependencyDecoder

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
        
    """
    # @staticmethod
    # def add_args(parser):
    #     """Add task-specific arguments to the parser."""
    #     print("add_args")
    #     parser.add_argument("data", help="path to data directory")
    #     parser.add_argument(
    #         "--tokens-per-sample",
    #         default=64,
    #         type=int,
    #         help="max number of total tokens over all segments"
    #         " per sample for dataset",
    #     )
    #     parser.add_argument(
    #         "--sample-break-mode",
    #         default="eos",
    #         type=str,
    #         help="mode for breaking sentence",
    #     )
    #     parser.add_argument("--dependency",
    #                     type=str,
    #                     default="/home/yangzhixian/DependencyGuided/DG/data/news/dependency",
    #                     help="path to data dependency")
    #     parser.add_argument("--dependency_suffix",
    #                     type=str,
    #                     default=".head",
    #                     help="file suffix of dependency head file")
    def __init__(self, args, dictionary, output_dictionary=None,):
        super().__init__(args)
        self.dictionary = dictionary
        self.output_dictionary = output_dictionary or dictionary
    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        dictionary = None
        output_dictionary = None
        if args.data:
            paths = utils.split_paths(args.data)
            assert len(paths) > 0
            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
            logger.info("dictionary: {} types".format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(
                    dictionary, args.output_dictionary_size
                )
        return (dictionary, output_dictionary)
    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary, output_dictionary = cls.setup_dictionary(args, **kwargs)

        return cls(args, dictionary, output_dictionary)
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = utils.split_paths(self.args.data)
        dependency_paths = utils.split_paths(self.args.dependency)
        assert len(paths) > 0
        assert len(dependency_paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        data_dependency_path = dependency_paths[(epoch - 1) % len(dependency_paths)]

        split_path = os.path.join(data_path, split)
        split_dependency_path = os.path.join(data_dependency_path, split)
        # each process has its own copy of the raw data (likely to be an np.memmap)
        dataset = data_utils.load_indexed_dataset(
            split_path, self.dictionary, self.args.dataset_impl, combine=combine
        )

        dataset_dependency = []
        with open(split_dependency_path + self.args.dependency_suffix, "r", encoding="utf-8") as f:
            for line in f:
                dataset_dependency.append(eval(line.strip("\n")))

        if dataset is None:
            raise FileNotFoundError(f"Dataset not found: {split} ({split_path})")
        if len(dataset_dependency) == 0:
            raise FileNotFoundError(f"Dataset dependency not found: {split} ({split_dependency_path + self.args.dependency_suffix})")

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )

        dataset = SentenceWithDependencyDataset(
            dataset,
            dataset.sizes,
            dataset_dependency,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            use_plasma_view=self.args.use_plasma_view,
            split_path=split_path,
            plasma_path=self.args.plasma_path,
        )

        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )

        self.datasets[split] = WrapDependencyDataset(
            dataset=dataset,
            sizes=dataset.sizes,
            src_vocab=self.dictionary,
            tgt_vocab=self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
        )
        
    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        raise NotImplementedError
    def inference_step(self, generator, models, sample, prefix_tokens=None):
        raise NotImplementedError
    def eval_lm_dataloader(
        self,
        dataset,
        max_tokens: Optional[int] = 36000,
        batch_size: Optional[int] = None,
        max_positions: Optional[int] = None,
        num_shards: int = 1,
        shard_id: int = 0,
        num_workers: int = 1,
        data_buffer_size: int = 10,
        # ensures that every evaluated token has access to a context of at least
        # this size, if possible
        context_window: int = 0,
    ):
        raise NotImplementedError

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.output_dictionary
    # def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
    #     """
    #     Generate batches for inference. We prepend an eos token to src_tokens
    #     (or bos if `--add-bos-token` is set) and we append a <pad> to target.
    #     This is convenient both for generation with a prefix and LM scoring.
    #     """
    #     dataset = StripTokenDataset(
    #         TokenBlockDataset(
    #             src_tokens,
    #             src_lengths,
    #             block_size=None,  # ignored for "eos" break mode
    #             pad=self.source_dictionary.pad(),
    #             eos=self.source_dictionary.eos(),
    #             break_mode="eos",
    #         ),
    #         # remove eos from (end of) target sequence
    #         self.source_dictionary.eos(),
    #     )
    #     src_dataset = PrependTokenDataset(
    #         dataset,
    #         token=(
    #             self.source_dictionary.bos()
    #             if getattr(self.args, "add_bos_token", False)
    #             else self.source_dictionary.eos()
    #         ),
    #     )
    