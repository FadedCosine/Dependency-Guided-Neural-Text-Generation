from dataclasses import dataclass, field
from typing import Optional

from fairseq import options, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (
    TransformerLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture,
)
from fairseq.models.transformer_lm import (
    TransformerLanguageModel,
    DEFAULT_MAX_TARGET_POSITIONS,
)
from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder
from omegaconf import II


@register_model("dependency_guided_transformer_pointer_generator")
class DGTransformerLM(TransformerLanguageModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        TransformerModel.add_args(parser)
        parser.add_argument('--alignment-heads', type=int, metavar='N',
                            help='number of attention heads to be used for '
                                 'pointing')
        parser.add_argument('--alignment-layer', type=int, metavar='I',
                            help='layer number to be used for pointing (0 '
                                 'corresponding to the bottommost layer)')
        parser.add_argument('--dependency-model-path', type=str, default="/home/yangzhixian/DependencyGuided/DG/checkpoints/new/dependency_predictor",
                            help='model path of dependency model')
        parser.add_argument('--dependency-model-filename', type=str, default="checkpoint_best.pt",
                            help='model filename of dependency model')
        parser.add_argument('--force-generation', type=float, metavar='P',
                            default=None,
                            help='set the vocabulary distribution weight to P, '
                                 'instead of predicting it from the input (1.0 '
                                 'corresponding to generation, 0.0 to pointing)')

    def __init__(self, decoder):
        super().__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
    
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
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)
        # In the pointer-generator model these arguments define the decoder
        # layer and the number of attention heads that will be averaged to
        # create the alignment for pointing.
        self.alignment_heads = args.alignment_heads
        self.alignment_layer = args.alignment_layer
        input_embed_dim = embed_tokens.embedding_dim
        self.dependency_decoder = TransformerLanguageModel.from_pretrained(
                args.dependency_model_path,
                checkpoint_file=args.dependency_model_filename,
                data_name_or_path=args.data,
                )
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
        tok_x, tok_extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=self.alignment_layer,
            alignment_heads=self.alignment_heads,
        )

        dep_x, dep_extra = self.dependency_decoder.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=self.alignment_layer,
            alignment_heads=self.alignment_heads,
        )
        # features_only is always false
        if not features_only:
            if incremental_state is not None:
                prev_output_tokens = prev_output_tokens[:, -1:]
            prev_output_embed = self.embed_tokens(prev_output_tokens)
            prev_output_embed *= self.embed_scale
            predictors = torch.cat((prev_output_embed, tok_x, dep_x), 2)
            p_gens = self.project_p_gens(predictors)
            p_gens = torch.sigmoid(p_gens)
            dep_attn: Optional[Tensor] = dep_extra["attn"][0]
            assert attn is not None
            tok_x = self.output_layer(tok_x, dep_x, attn, encoder_out["src_tokens"][0], p_gens)

    def output_layer(
        self,
        tok_features: Tensor,
        dep_features: Tensor,
        dep_attn: Tensor,
        dep_loss_type: str,
        p_gens: Tensor
    ) -> Tensor:
        """
        Project dependency attention features to the weight of dependency probabilities
        """
        if self.force_p_gen is not None:
            p_gens = self.force_p_gen

        if self.adaptive_softmax is None:
            next_token_logits = self.output_projection(tok_features)
        else:
            next_token_logits = tok_features

        if self.dependency_decoder.adaptive_softmax is None:
            dep_token_logits = self.dependency_decoder.output_projection(dep_features)
        else:
            dep_token_logits = dep_features

        batch_size = next_token_logits.shape[0]
        output_length = next_token_logits.shape[1]

        gen_dists = self.get_normalized_probs_scriptable(
            (next_token_logits, None), log_probs=False, sample=None
        )
        gen_dists = torch.mul(gen_dists, p_gens)

        assert dep_attn.shape[2] == dep_token_logits.shape[1]
        weighted_sum_dep_dists = torch.bmm(dep_attn,dep_token_logits)
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
