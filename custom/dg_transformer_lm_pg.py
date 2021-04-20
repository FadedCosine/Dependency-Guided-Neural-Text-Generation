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
    the output probabilities with an dependency attention distribution weighted dependency probabilities in the output layer.

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
