from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    TransformerEncoder,
    TransformerDecoder,
    TransformerModel,
    DEFAULT_MIN_PARAMS_TO_WRAP, Embedding,
    base_architecture,
)
from torch import Tensor

@register_model("dependency_pointer_transformer")
class DPTransformer(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True
        self.is_train_dependency = args.is_train_dependency
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        TransformerModel.add_args(parser)
        parser.add_argument('--alignment-heads', type=int, default=1,
                            help='number of attention heads to be used for '
                                 'pointing')
        parser.add_argument('--alignment-layer', type=int, default=-1,
                            help='layer number to be used for pointing (0 '
                                 'corresponding to the bottommost layer)')
        
        parser.add_argument('--force-generation', type=float, metavar='P',
                            default=-1,
                            help='set the vocabulary distribution weight to P, '
                                 'instead of predicting it from the input (1.0 '
                                 'corresponding to generation, 0.0 to pointing)')
        parser.add_argument('--freeze-dependency-decoder', action="store_true",
                                help="If freeze dependency decoder")
        parser.add_argument('--is-train-dependency', action="store_true",
                                help="If is training dependency decoder")
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))
        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP
            )
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)
        return cls(args, encoder, decoder)
    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return DGTransformerPointerGeneratorDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # In DPTransformer, nexttoken_decoder and dependency_decoder share the same encoder
        encoder_out = self.encoder(
                src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
            )
        # the code in DGTransformerPointerGeneratorDecoder will help to train dependency_decoder and nexttoken_decoder, respectively
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out
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

class DGTransformerPointerGeneratorDecoder(FairseqIncrementalDecoder):
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
        # In the pointer-generator model these arguments define the decoder
        # layer and the number of attention heads that will be averaged to
        # create the alignment for pointing.
        super().__init__(dictionary)
        self.alignment_heads = args.alignment_heads
        self.alignment_layer = args.alignment_layer
        input_embed_dim = embed_tokens.embedding_dim
        assert args.dependency_model_path is not None
        assert args.dependency_model_filename is not None
        
        self.nexttoken_decoder = TransformerDecoder(
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
        )
        self.dependency_decoder = TransformerDecoder(
            args,
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
        )
        # freeze pretrained model
        #! 不丑陋地直接在init中加载模型，但还是根据参数控制dependency decoder是否进行训练，不知可行否
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
        self.is_train_dependency = args.is_train_dependency
    
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
        
        if self.is_train_dependency:
            dep_x_all_output, dep_extra = self.dependency_decoder.extract_features(
                prev_output_tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_state, # when training, incremental_state is None
                alignment_layer=self.alignment_layer,
                alignment_heads=self.alignment_heads,
            )
            if not features_only:
                dep_x = self.output_layer(None, dep_x_all_output, None, None)
            return dep_x, dep_extra
        else:
            tok_x, tok_extra = self.nexttoken_decoder.extract_features(
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
                else:
                    dep_x_cur = dep_x_all_output

                prev_output_embed = self.nexttoken_decoder.embed_tokens(prev_output_tokens)
                prev_output_embed *= self.nexttoken_decoder.embed_scale
            
                predictors = torch.cat((prev_output_embed, tok_x, dep_x_cur), 2)
                p_gens = self.project_p_gens(predictors)
                p_gens = torch.sigmoid(p_gens)
                dep_attn: Optional[Tensor] = dep_extra["attn"][0]
                
                assert dep_attn is not None
                tok_x = self.output_layer(tok_x, dep_x_all_output, dep_attn, p_gens)
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
        output for dependency decoder only or
        Project dependency attention features to the weight of dependency probabilities
        
        The return is the truth probabilities, not logits.
        """
        if tok_features is None:
            if self.dependency_decoder.decoder.adaptive_softmax is None:
                dep_token_logits = self.dependency_decoder.decoder.output_projection(dep_features)
            else:
                dep_token_logits = dep_features
            dep_token_logits = self.dependency_decoder.get_normalized_probs_scriptable(
                (dep_token_logits, None), log_probs=False, sample=None
            )
            return dep_token_logits
        else:
            if self.force_p_gen > 0:
                p_gens = self.force_p_gen

            if self.nexttoken_decoder.adaptive_softmax is None:
                next_token_logits = self.nexttoken_decoder.output_projection(tok_features)
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

            gen_dists = self.nexttoken_decoder.get_normalized_probs_scriptable(
                (next_token_logits, None), log_probs=False, sample=None
            )
            gen_dists = torch.mul(gen_dists, p_gens)
            # print("dep_attn size is : ", dep_attn.size())
            # print("dep_token_logits size is : ", dep_token_logits.size())
            assert dep_attn.shape[2] == dep_token_logits.shape[1]
            weighted_sum_dep_dists = torch.bmm(dep_attn, dep_token_logits)
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


@register_model_architecture(
    "dependency_pointer_transformer", "dependency_pointer_transformer"
)
def dependency_pointer_transformer(args):
    args.alignment_heads = getattr(args, "alignment_heads", 1)
    args.alignment_layer = getattr(args, "alignment_layer", -1)
    base_architecture(args)
    if args.alignment_layer < 0:
        args.alignment_layer = args.decoder_layers + args.alignment_layer
    args.force_generation = getattr(args, "force_generation", -1)
    args.freeze_dependency_decoder = getattr(args, "freeze_dependency_decoder", True)
    args.is_train_dependency = getattr(args, "is_train_dependency", True)
