from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple

from fairseq import options, utils, hub_utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer_lm import TransformerLanguageModel
from fairseq.models.transformer import (
    TransformerDecoder,
    DEFAULT_MIN_PARAMS_TO_WRAP, Embedding,
    base_architecture,
)
from fairseq.models.transformer_lm import (
    DEFAULT_MAX_TARGET_POSITIONS,
)
from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder
from omegaconf import II

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class DPTransformerLanguageModelConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    relu_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    decoder_embed_dim: int = field(
        default=512, metadata={"help": "decoder embedding dimension"}
    )
    decoder_output_dim: int = field(
        default=512, metadata={"help": "decoder output dimension"}
    )
    decoder_input_dim: int = field(
        default=512, metadata={"help": "decoder input dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num decoder layers"})
    decoder_attention_heads: int = field(
        default=8, metadata={"help": "num decoder attention heads"}
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_decoder_final_norm: bool = field(
        default=False,
        metadata={"help": "don't add an extra layernorm after the last decoder block"},
    )
    adaptive_softmax_cutoff: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated list of adaptive softmax cutoff points. "
            "Must be used with adaptive_loss criterion"
        },
    )
    adaptive_softmax_dropout: float = field(
        default=0,
        metadata={"help": "sets adaptive softmax dropout for the tail projections"},
    )
    adaptive_softmax_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    character_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, uses character embedding convolutions to produce token embeddings"
        },
    )
    character_filters: str = field(
        default="[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]",
        metadata={"help": "size of character embeddings"},
    )
    character_embedding_dim: int = field(
        default=4, metadata={"help": "size of character embeddings"}
    )
    char_embedder_highway_layers: int = field(
        default=2,
        metadata={"help": "number of highway layers for character token embeddder"},
    )
    adaptive_input: bool = field(
        default=False, metadata={"help": "if set, uses adaptive input"}
    )
    adaptive_input_factor: float = field(
        default=4, metadata={"help": "adaptive input factor"}
    )
    adaptive_input_cutoff: Optional[str] = field(
        default=None,
        metadata={"help": "comma separated list of adaptive input cutoff points."},
    )
    tie_adaptive_weights: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the weights of adaptive softmax and adaptive input"
        },
    )
    tie_adaptive_proj: bool = field(
        default=False,
        metadata={
            "help": "if set, ties the projection weights of adaptive softmax and adaptive input"
        },
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    # config for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for decoder"}
    )
    decoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "which layers to *keep* when pruning as a comma-separated list"
        },
    )
    # config for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    quant_noise_pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    quant_noise_scalar: float = field(
        default=0.0,
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )
    # config for Fully Sharded Data Parallel (FSDP) training
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": (
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        }
    )
    # options from other parts of the config
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    tpu: bool = II("common.tpu")

    # for dependency pointer
    data_path_for_load_model: str = field(
        default="/home/yangzhixian/DependencyGuided/data/news/data-bin",
        metadata={
            "help": "path to data"
        },
    )
    alignment_heads: int = field(
        default=1, metadata={"help": "number of attention heads to be used for pointing"}
    )
    alignment_layer: int = field(
        default=-1, metadata={"help": "layer number to be used for pointing (0 corresponding to the bottommost layer)"}
    )
    dependency_model_path: str = field(
        default="/home/yangzhixian/DependencyGuided/DG/checkpoints/news/dependency_predictor",
        metadata={
            "help": "model path of dependency decoder model"
        },
    )
    dependency_model_filename: str = field(
        default="checkpoint_best.pt",
        metadata={
            "help": "model filename of dependency decoder model"
        },
    )
    force_generation: float = field(
        default=0,
        metadata={
            "help": 'set the vocabulary distribution weight to P, '
                    'instead of predicting it from the input (1.0 '
                    'corresponding to generation, 0.0 to pointing)'
        },
    )
    freeze_dependency_decoder: bool = field(
        default=False,
        metadata={
            "help": 'if freeze the dependency decoder while training DPTransformerLM'
        },
    )
    

@register_model("dependency_pointer_transformer_lm", dataclass=DPTransformerLanguageModelConfig)
class DPTransformerLM(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_lm_architecture(args)
        
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary,
                eval(args.character_filters),
                args.character_embedding_dim,
                args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary),
                task.source_dictionary.pad(),
                args.decoder_input_dim,
                args.adaptive_input_factor,
                args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            embed_tokens = cls.build_embedding(
                args, task.source_dictionary, args.decoder_input_dim
            )

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert (
                args.adaptive_softmax_cutoff == args.adaptive_input_cutoff
            ), "{} != {}".format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff
            )
            assert args.decoder_input_dim == args.decoder_output_dim

        decoder = DGTransformerPointerGeneratorDecoder(args, task.target_dictionary, embed_tokens, no_encoder_attn=True)
        # decoder = TransformerDecoder(
        #     args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        # )
        return cls(decoder)
    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        embed_tokens = Embedding(len(dictionary), embed_dim, dictionary.pad())
        return embed_tokens
    

class DGTransformerPointerGeneratorDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`. 
    The DG variant means there is a Dependency Decoder with fixed parameters inside this Transformer decoder. The pointer-generator variant mixes
    the output probabilities with an dependency attention weighted dependency probabilities in the output layer.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
    """
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        # In the pointer-generator model these arguments define the decoder
        # layer and the number of attention heads that will be averaged to
        # create the alignment for pointing.
        self.alignment_heads = args.alignment_heads
        self.alignment_layer = args.alignment_layer
        input_embed_dim = embed_tokens.embedding_dim
        assert args.dependency_model_path is not None
        assert args.dependency_model_filename is not None
        # print("args.dependency_model_path is : ", args.dependency_model_path)
        # print("args.dependency_model_filename is : ", args.dependency_model_filename)
        # print("args.data_path_for_load_model is : ", args.data_path_for_load_model)
        self.dependency_module = hub_utils.from_pretrained(
            args.dependency_model_path,
            args.dependency_model_filename,
            args.data_path_for_load_model,
        )
        self.dependency_decoder = self.dependency_module["models"][0]
        # freeze pretrained model
        if args.freeze_dependency_decoder:
            for param in self.dependency_decoder.parameters():
                param.requires_grad = False
        else:
            for param in self.dependency_decoder.parameters():
                param.requires_grad = True
        # Generation probabilities / interpolation coefficients are predicted
        # from the both decoders input embedding (the same), the token decoder output, and the dependency decoder output
        
        p_gen_input_size = input_embed_dim + self.output_embed_dim + self.dependency_decoder.decoder.output_embed_dim
        self.project_p_gens = nn.Linear(p_gen_input_size, 1)
        nn.init.zeros_(self.project_p_gens.bias)

        self.force_p_gen = args.force_generation
    
    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = 0,
        alignment_heads: Optional[int] = 1,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict, optional): dictionary used for storing
                state during :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False)
            alignment_layer (int, optional): 0-based index of the layer to be
                used for pointing (default: 0)
            alignment_heads (int, optional): number of attention heads to be
                used for pointing (default: 1)

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        # The normal Transformer model doesn't pass the alignment_layer and
        # alignment_heads parameters correctly. We use our local variables.
        # print("prev_output_tokens size is : ", prev_output_tokens.size())
        
        tok_x, tok_extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=self.alignment_layer,
            alignment_heads=self.alignment_heads,
        )
        
        dep_x_all_output, dep_extra = self.dependency_decoder.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            # incremental_state=incremental_state, we can not to use the same incremental_state for dependency decoder, 
            # so a ugly way is to recalculate the attention of all the prev_output_tokens
            alignment_layer=self.alignment_layer,
            alignment_heads=self.alignment_heads,
        )
        # features_only is always false
        if not features_only:
            if incremental_state is not None: #means that we are in generating phase
                prev_output_tokens = prev_output_tokens[:, -1:]
                dep_x_cur = dep_x_all_output[:, -1:, :]
                dep_attn: Optional[Tensor] = dep_extra["attn"][0][:, -1:, :]
                # tok_attn: Optional[Tensor] = tok_extra["attn"][0][:, -1:, :]
            else:
                dep_x_cur = dep_x_all_output
                dep_attn: Optional[Tensor] = dep_extra["attn"][0]
                # tok_attn: Optional[Tensor] = tok_extra["attn"][0]
            # print("dep_x_cur is size : ", dep_x_cur.size())
            prev_output_embed = self.embed_tokens(prev_output_tokens)
            prev_output_embed *= self.embed_scale
           
            assert dep_attn.shape[2] == dep_x_all_output.shape[1]
            # assert tok_attn.shape[2] == dep_x_all_output.shape[1]
            # dep_context_state = torch.bmm(dep_attn, dep_x_all_output)
            # predictors = torch.cat((prev_output_embed, tok_x, dep_context_state), 2)
            """
            transformer 结构中的self-attn在decoder的t时刻的hidden state dep_x_cur，本身就包含着对于之前所有attn向量和其v值的加权求和
            """
            predictors = torch.cat((prev_output_embed, tok_x, dep_x_cur), 2)
            p_gens = self.project_p_gens(predictors)
            p_gens = torch.sigmoid(p_gens)
            assert dep_attn is not None
            tok_x = self.output_layer(tok_x, dep_x_all_output, dep_attn, p_gens)
            # assert tok_attn is not None
            # tok_x = self.output_layer(tok_x, dep_x_all_output, tok_attn, p_gens)
        return tok_x, tok_extra

    def output_layer(
        self,
        tok_features: Tensor,
        dep_features: Tensor,
        dep_attn: Tensor,
        # dep_loss_type: str,
        p_gens: Tensor
    ) -> Tensor:
        """
        Project dependency attention features to the weight of dependency probabilities
        """
        if self.force_p_gen >= 0:
            p_gens = self.force_p_gen
        # print("p_gens is : ", p_gens)
        if self.adaptive_softmax is None:
            next_token_logits = self.output_projection(tok_features)
        else:
            next_token_logits = tok_features

        if self.dependency_decoder.decoder.adaptive_softmax is None:
            dep_token_logits = self.dependency_decoder.decoder.output_projection(dep_features)
        else:
            dep_token_logits = dep_features

        # if dep_loss_type == "CE":
        dep_token_logits = self.dependency_decoder.get_normalized_probs_scriptable(
            (dep_token_logits, None), log_probs=False, sample=None
        )
        # elif dep_loss_type == "BCE":

        gen_dists = self.get_normalized_probs_scriptable(
            (next_token_logits, None), log_probs=False, sample=None
        )
        gen_dists = torch.mul(gen_dists, p_gens)
        # print("gen_dists size is : ", gen_dists.size())
        assert dep_attn.shape[2] == dep_token_logits.shape[1]
        # print("torch.sum(dep_attn, -1) is : ", torch.sum(dep_attn, -1))
        weighted_sum_dep_dists = torch.bmm(dep_attn, dep_token_logits)
        # print("dep_token_logits size is : ", dep_token_logits.size())
        # print("weighted_sum_dep_dists size is : ", weighted_sum_dep_dists.size())
        weighted_sum_dep_dists = torch.mul(weighted_sum_dep_dists, 1 - p_gens)

        return gen_dists + weighted_sum_dep_dists
    def get_normalized_probs(
        self, 
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """
        Get normalized probabilities (or log probs) from a net's output.
        Pointer-generator network output is already normalized.
        """
        probs = net_output[0]
        # Make sure the probabilities are greater than zero when returning log
        # probabilities.
        return probs.clamp(1e-10, 1.0).log() if log_probs else probs




def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, "no_tie_adaptive_proj"):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, "decoder_final_norm"):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, "adaptive_softmax_factor", 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.activation_fn = getattr(args, "activation_fn", "relu")

    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.character_embeddings = getattr(args, "character_embeddings", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", False)

    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.adaptive_input_factor = getattr(args, "adaptive_input_factor", 4)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", None)

    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.data_path_for_load_model = getattr(args, "data_path_for_load_model", "/home/yangzhixian/DependencyGuided/data/news/data-bin")
    args.alignment_heads = getattr(args, "alignment_heads", 1)
    args.alignment_layer = getattr(args, "alignment_layer", -1)
    if args.alignment_layer < 0:
        args.alignment_layer = args.decoder_layers + args.alignment_layer
    args.dependency_model_path = getattr(args, "dependency_model_path", "/home/yangzhixian/DependencyGuided/DG/checkpoints/news/dependency_predictor")
    args.dependency_model_filename = getattr(args, "dependency_model_filename", "checkpoint_best.pt")
    args.force_generation = getattr(args, "force_generation", 0)
    args.freeze_dependency_decoder = getattr(args, "freeze_dependency_decoder", False)
    

@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_big")
def dependency_pointer_transformer_lm_big(args):
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_lm_architecture(args)

@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_wiki103")
@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_baevski_wiki103")
def dependency_pointer_transformer_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, "decoder_layers", 16)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.3)
    args.adaptive_input = getattr(args, "adaptive_input", True)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", True)
    args.adaptive_input_cutoff = getattr(args, "adaptive_input_cutoff", "20000,60000")
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", "20000,60000"
    )
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0.2)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    args.tie_adaptive_proj = getattr(args, "tie_adaptive_proj", True)
    dependency_pointer_transformer_lm_big(args)


@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_gbw")
@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_baevski_gbw")
def dependency_pointer_transformer_lm_baevski_gbw(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.no_decoder_final_norm = getattr(args, "no_decoder_final_norm", True)
    dependency_pointer_transformer_lm_big(args)


@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_gpt")
def dependency_pointer_transformer_lm_gpt(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_gpt2_small")
def dependency_pointer_transformer_lm_gpt2_small(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_gpt2_tiny")
def dependency_pointer_transformer_lm_gpt2_tiny(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 64)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 64)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 1)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_gpt2_medium")
def dependency_pointer_transformer_lm_gpt2_medium(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1280)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 5120)
    args.decoder_layers = getattr(args, "decoder_layers", 36)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 20)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_gpt2_big")
def dependency_pointer_transformer_lm_gpt2_big(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1600)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 6400)
    args.decoder_layers = getattr(args, "decoder_layers", 48)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 25)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


def base_gpt3_architecture(args):
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.decoder_embed_dim * 4)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_gpt3_small")
def dependency_pointer_transformer_lm_gpt3_small(args):
    # 125M params
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    base_gpt3_architecture(args)


@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_gpt3_medium")
def dependency_pointer_transformer_lm_gpt3_medium(args):
    # 350M params
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_gpt3_architecture(args)


@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_gpt3_large")
def dependency_pointer_transformer_lm_gpt3_large(args):
    # 760M params
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1536)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_gpt3_architecture(args)


@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_gpt3_xl")
def dependency_pointer_transformer_lm_gpt3_xl(args):
    # 1.3B params
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 24)
    base_gpt3_architecture(args)


@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_gpt3_2_7")
def dependency_pointer_transformer_lm_gpt3_2_7(args):
    # 2.7B params
    args.decoder_layers = getattr(args, "decoder_layers", 32)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2560)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    base_gpt3_architecture(args)


@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_gpt3_6_7")
def dependency_pointer_transformer_lm_gpt3_6_7(args):
    # 6.7B params
    args.decoder_layers = getattr(args, "decoder_layers", 32)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    base_gpt3_architecture(args)


@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_gpt3_13")
def dependency_pointer_transformer_lm_gpt3_13(args):
    # 13B params
    args.decoder_layers = getattr(args, "decoder_layers", 40)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 5120)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 40)
    base_gpt3_architecture(args)


@register_model_architecture("dependency_pointer_transformer_lm", "dependency_pointer_transformer_lm_gpt3_175")
def dependency_pointer_transformer_lm_gpt3_175(args):
    # 175B params
    args.decoder_layers = getattr(args, "decoder_layers", 96)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 12288)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 96)
    base_gpt3_architecture(args)

