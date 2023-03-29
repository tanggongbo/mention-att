import logging

import fairseq
from torch import nn
from torch.nn import functional as F

from fairseq.models import FairseqEncoder, FairseqDecoder
from fairseq.models.fairseq_model import BaseFairseqModel
from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder, EncoderOut, DEFAULT_MAX_SOURCE_POSITIONS, DEFAULT_MAX_TARGET_POSITIONS
from fairseq import utils
import copy
import logging
from collections import OrderedDict, namedtuple
from typing import Union

from fairseq.file_io import PathManager
from fairseq import checkpoint_utils

from .utils import *
from .mention_decoder import TransformerDecoder4Mention

ModelOut = namedtuple('ModelOut', [
    'decoder_out',  # the (decoder out, extra), same as original,
    'encoder_ne_logit',
    'decoder_ne_logit',
    'entity_out',
    'entity_label',
    'result_entity_id',
    'encoder_ne'
])

NE_PENALTY = 1e8
logger = logging.getLogger(__name__)

class EntityEncoderDecoderModel(BaseFairseqModel):

    def __init__(self, args, ne_dict, encoder, decoder, decoder4mention, tgt_ne_start_id):
        super().__init__()

        self.args = args
        self.ne_dict = ne_dict
        self.encoder = encoder
        self.decoder = decoder
        self.decoder4mention = decoder4mention
        self.mode = args.mode
        self.tgt_ne_start_id = tgt_ne_start_id

        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, FairseqDecoder)
        assert isinstance(self.decoder4mention, FairseqDecoder)
        assert 0 <= self.mode <= 3
        assert 0 <= self.args.bert_lookup_layer <= self.args.decoder_layers # 0 is the input
        assert 1 <= self.args.src_ne_layer <= self.args.encoder_layers

        self.encoder_ne_process_mask = {}
        self.src_ne_fc1 = nn.Linear(args.encoder_embed_dim, args.src_ne_project, bias=True)
        self.src_ne_fc2 = nn.Linear(args.src_ne_project, len(ne_dict), bias=True)
        self.tgt_ne_fc1 = nn.Linear(args.decoder_embed_dim, args.tgt_ne_project, bias=True)
        self.tgt_ne_fc2 = nn.Linear(args.tgt_ne_project, len(ne_dict), bias=True)

        self.mention_tag_src = None
        self.mention_tag_trg = None
        self.mention_tag_idx = self.ne_dict.index('mention')  # the index of "mention" tag

    @staticmethod
    def add_args(parser):
        parser.add_argument('--src-ne-project', type=int)
        parser.add_argument('--src-ne-project-dropout', type=float, default=0.0)
        parser.add_argument('--tgt-ne-project', type=int)
        parser.add_argument('--concat-ne-emb', action='store_true')
        parser.add_argument('--bert-lookup-layer', type=int) # 0 is the input, and n is the last layer
        parser.add_argument('--bert-lookup-dropout', type=float, default=0.0) #
        parser.add_argument('--src-ne-layer', type=int) # 1 is the first layer output, and n is the last layer
        parser.add_argument('--decoder-mention-layer', type=int)  # the number of layers in decoder4mention
        parser.add_argument('--mention-attention-layer', type=int)  # the layer to apply mention attention
        parser.add_argument('--ratio-mention-ctx', type=float, default=1.0)  # the ratio of ctx vector from mention attn
        parser.add_argument('--gold-label', action='store_true')
        parser.add_argument('--no-mention-mask', action='store_true')
        parser.add_argument('--pretrain', type=str)

    def init_tags(self, src_tag, trg_tag):
        self.mention_tag_src = src_tag
        self.mention_tag_trg = trg_tag

    #@profile
    def encoder_ne_process(self, encoder_out, need_logit):

        if encoder_out.encoder_states is None:
            entity_input = encoder_out.encoder_out
        else:
            assert len(encoder_out.encoder_states) == self.args.encoder_layers
            entity_input = encoder_out.encoder_states[self.args.src_ne_layer - 1] # T B C
        encoder_ne_emb = F.dropout(F.relu(self.src_ne_fc1(entity_input)), self.args.src_ne_project_dropout, self.training)

        if need_logit:
            encoder_ne_logit = self.src_ne_fc2(encoder_ne_emb)
            encoder_ne_logit = encoder_ne_logit.transpose(0, 1)  # T x B x C => B x T x C
        else:
            encoder_ne_logit = None

        if self.args.concat_ne_emb:
            combined_encoder_out = torch.cat((encoder_out.encoder_out, encoder_ne_emb), dim=-1) # T x B x C
        else:
            combined_encoder_out = encoder_out.encoder_out + encoder_ne_emb
        encoder_out_with_emb = EncoderOut(
            encoder_out=combined_encoder_out,
            encoder_padding_mask=encoder_out.encoder_padding_mask,  # B x T
            encoder_embedding=encoder_out.encoder_embedding,  # B x T x C
            encoder_states=encoder_out.encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

        return encoder_out_with_emb, encoder_ne_logit
    
    #@profile
    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
                - the encoder ne logit
                - the decoder ne logit
        """

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=(self.mode != 0), **kwargs)
        # print("src_tokens.shape", src_tokens.shape)
        # print("encoder_out.encoder_states[self.args.src_ne_layer - 1].shape", encoder_out.encoder_states[self.args.src_ne_layer - 1].shape)

        encoder_out_with_emb, encoder_ne_logit = self.encoder_ne_process(encoder_out=encoder_out, need_logit=True)
        if self.training:
            # 1e-8/1e-4 for masks, following fairseq source code
            '''transformer_layer.html#TransformerEncoderLayer'''
            if not self.args.no_mention_mask:
                # (bsz, src_len)
                mention_mask = self.mention_tag_src.new_zeros(self.mention_tag_src.size())
                if self.mention_tag_src != None and self.mention_tag_trg != None:
                    mention_mask = mention_mask.masked_fill(self.mention_tag_src != self.mention_tag_idx,
                                                            -1e8 if encoder_out.encoder_out.dtype == torch.float32 else -1e4)
            else:
                mention_mask = None

            # decoder_out_feature: B x T x C
            decoder_out_feature, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out_with_emb,
                                                      features_only=True, **kwargs)

            assert (decoder_out_feature == extra['inner_states'][-1].transpose(0, 1)).all()

            ne_input_feature = extra['inner_states'][self.args.bert_lookup_layer].transpose(0, 1)

            decoder_ne_emb = F.relu(self.tgt_ne_fc1(ne_input_feature))
            decoder_ne_logit = self.tgt_ne_fc2(decoder_ne_emb)

            # pass the conventional decoder outputs and src-side mention masks to the decoder4mention

            self.decoder4mention.set_input_and_mask(decoder_out_feature, mention_mask)

            decoder_out_feature_mention, extra = self.decoder4mention(prev_output_tokens, encoder_out=encoder_out_with_emb,
                                                              features_only=True, **kwargs)

            # print(self.mention_tag_trg)
            mention_or_not = (self.mention_tag_trg ==
                              self.mention_tag_idx).unsqueeze(dim=-1).repeat(1, 1, decoder_out_feature.size()[-1])
            # print(mention_or_not.any())
            # decoder_out_feature_final = torch.where(mention_or_not, decoder_out_feature_mention, decoder_out_feature)
            # decoder_out_feature_final = torch.where(mention_or_not, decoder_out_feature_mention
            #                                         + self.args.ratio_mention_ctx * decoder_out_feature,
            #                                         decoder_out_feature)
            # print((decoder_out_feature_final_2 == decoder_out_feature_final).all())

            # decoder_out_feature_final = decoder_out_feature
            decoder_out_feature_final = decoder_out_feature_mention

            if self.decoder.share_input_output_embed:
                decoder_out = F.linear(decoder_out_feature_final, self.decoder.embed_tokens.weight)
            else:
                decoder_out = F.linear(decoder_out_feature_final, self.decoder.output_projection.weight)

            return ModelOut(
                decoder_out=(decoder_out, extra),
                encoder_ne_logit=encoder_ne_logit,
                decoder_ne_logit=decoder_ne_logit,
                entity_out=None,
                entity_label=None,
                result_entity_id=None,
                encoder_ne=None
            )
        else:
            # different code for validation
            model_out = self._forward_decoder(prev_output_tokens, encoder_out, **kwargs)
            return model_out

    def _forward_decoder(self, prev_output_tokens, encoder_out, **kwargs):
        encoder_out_with_emb, encoder_ne_logit = self.encoder_ne_process(encoder_out, need_logit=True)
        # decoder_out_feature: B x T x C
        decoder_out_feature, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out_with_emb,
                                                  features_only=True, **kwargs)
        assert (decoder_out_feature == extra['inner_states'][-1].transpose(0, 1)).all()
        ne_input_feature = extra['inner_states'][self.args.bert_lookup_layer].transpose(0, 1)
        decoder_ne_emb = F.relu(self.tgt_ne_fc1(ne_input_feature))
        decoder_ne_logit = self.tgt_ne_fc2(decoder_ne_emb)

        # get the mention tags from predictions
        mention_tag_src = encoder_ne_logit.argmax(dim=-1)  # BxTxC->BxT
        # print(f'mention_tag_src: {mention_tag_src}')
        if not self.args.no_mention_mask:

            mention_mask = mention_tag_src.new_zeros(mention_tag_src.size())
            if mention_tag_src != None:
                mention_mask = mention_mask.masked_fill(mention_tag_src != self.mention_tag_idx,
                                                        -1e8 if encoder_out.encoder_out.dtype == torch.float32 else -1e4)
        else:
            mention_mask = None
        # pass the conventional decoder outputs and src-side mention masks to the decoder4mention
        self.decoder4mention.set_input_and_mask(decoder_out_feature, mention_mask)
        # print("start to process the mention decoder")
        decoder_out_feature_mention, extra = self.decoder4mention(prev_output_tokens, encoder_out=encoder_out_with_emb,
                                                                  features_only=True, **kwargs)
        # get the mention tags from predictions
        mention_tag_trg = decoder_ne_logit.argmax(dim=-1)
        # print(f'mention_tag_trg: {mention_tag_trg}')
        mention_or_not = (mention_tag_trg == self.mention_tag_idx).unsqueeze(dim=-1).repeat(
            1, 1, decoder_out_feature.size()[-1])
        # mix decoder_output_feature and decoder_output_feature_mention
        # decoder_out_feature_final = torch.where(mention_or_not, decoder_out_feature_mention, decoder_out_feature)
        # decoder_out_feature_final = torch.where(mention_or_not, decoder_out_feature_mention
        #                                         + self.args.ratio_mention_ctx * decoder_out_feature,
        #                                         decoder_out_feature)
        # decoder_out_feature_final = decoder_out_feature
        decoder_out_feature_final = decoder_out_feature_mention

        if self.decoder.share_input_output_embed:
            decoder_out = F.linear(decoder_out_feature_final, self.decoder.embed_tokens.weight)
        else:
            decoder_out = F.linear(decoder_out_feature_final, self.decoder.output_projection.weight)

        return ModelOut(
            decoder_out=(decoder_out, extra),
            encoder_ne_logit=encoder_ne_logit,
            decoder_ne_logit=decoder_ne_logit,
            entity_out=None,
            entity_label=None,
            result_entity_id=None,
            encoder_ne=None
        )

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = self.decoder.extract_features(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()


@fairseq.models.register_model('entity_transformer')
class EntityTransformer(EntityEncoderDecoderModel):

    def __init__(self, args, ne_dict, encoder, decoder, decoder4mention, tgt_ne_start_id):
        super().__init__(args, ne_dict, encoder, decoder, decoder4mention, tgt_ne_start_id)
        self.args = args

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        EntityEncoderDecoderModel.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        ##### Copy from transformer.py ####
        # make sure all arguments are present in older models
        base_architecture(args)

        if getattr(args, 'encoder_layers_to_keep', None):
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if getattr(args, 'decoder_layers_to_keep', None):
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, 'max_source_positions', None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = fairseq.models.transformer.Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        if args.pretrain:
            with torch.no_grad():
                encoder, encoder_embed_tokens, _ = load_pretrained_component_from_model(component=encoder,
                                                                                       embed=encoder_embed_tokens,
                                                                                       pretrain=args.pretrain,
                                                                                       checkpoint=args.pretrain)
                logger.info(f"loaded pretrained encoder from"
                            f"{args.pretrain}")

        if args.mode != 0 and args.concat_ne_emb:
            new_args = copy.deepcopy(args)
            new_args.encoder_embed_dim = args.encoder_embed_dim + args.src_ne_project
            decoder = TransformerDecoder(new_args, tgt_dict, decoder_embed_tokens)
            decoder4mention = TransformerDecoder4Mention(new_args, tgt_dict, decoder_embed_tokens)
        else:
            assert args.mode == 0 or args.encoder_embed_dim == args.src_ne_project, f'mode {args.mode}, {args.encoder_embed_dim} != {args.src_ne_project}'
            decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
            if args.pretrain:
                with torch.no_grad():
                    # update the embedding matrix and share with the decoder4mention
                    decoder, decoder_embed_tokens, pretrained_output_proj = load_pretrained_component_from_model(
                        component=decoder, embed=decoder_embed_tokens, pretrain=args.pretrain, checkpoint=args.pretrain)
                    logger.info(f"loaded pretrained decoder from"
                                f"{args.pretrain}")

            decoder4mention = TransformerDecoder4Mention(args, tgt_dict, decoder_embed_tokens, pretrained_output_proj)

        tgt_ne_start_id = len(task.tgt_dict.lang_dict)
        return cls(args, task.ne_dict, encoder, decoder, decoder4mention, tgt_ne_start_id)


@fairseq.models.register_model_architecture('entity_transformer', 'entity_transformer')
def base_architecture(args):
    fairseq.models.transformer.base_architecture(args)
    args.concat_ne_emb = getattr(args, 'concat_ne_emb', False)
    args.src_ne_project = getattr(args, 'src_ne_project', args.encoder_embed_dim)
    args.tgt_ne_project = getattr(args, 'tgt_ne_project', args.src_ne_project)
    args.bert_lookup_layer = getattr(args, 'bert_lookup_layer', args.decoder_layers) # Use last layer by default
    args.src_ne_layer = getattr(args, 'src_ne_layer', args.encoder_layers)
    args.tgt_ne_drop_rate = getattr(args, 'tgt_ne_drop_rate', 0.0)
    args.mode = getattr(args, 'mode', 1)
    args.decoder_mention_layer = getattr(args, 'decoder_mention_layer', 6)
    args.pretrain = getattr(args, 'pretrain', None)
    args.no_mention_mask = getattr(args, 'no_mention_mask', False)


@fairseq.models.register_model_architecture('entity_transformer', 'entity_transformer_iwslt_de_en')
def transformer_iwslt_de_en(args):
    fairseq.models.transformer.transformer_iwslt_de_en(args)
    base_architecture(args)


@fairseq.models.register_model_architecture('entity_transformer', 'entity_transformer_vaswani_wmt_en_de_big')
def transformer_vaswani_wmt_en_de_big(args):
    fairseq.models.transformer.transformer_vaswani_wmt_en_de_big(args)
    base_architecture(args)


@fairseq.models.register_model_architecture('entity_transformer', 'entity_transformer_wmt_en_de')
def transformer_wmt_en_de(args):
    fairseq.models.transformer.transformer_wmt_en_de(args)
    base_architecture(args)

def load_pretrained_component_from_model(
    component: Union[FairseqEncoder, FairseqDecoder],
    embed: fairseq.models.transformer.Embedding,
    pretrain: str,
    checkpoint: str
):
    """
    Load a pretrained FairseqEncoder or FairseqDecoder from checkpoint into the
    provided `component` object. If state_dict fails to load, there may be a
    mismatch in the architecture of the corresponding `component` found in the
    `checkpoint` file.
    """
    if not PathManager.exists(checkpoint):
        raise IOError("Model file not found: {}".format(checkpoint))
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint)
    if isinstance(component, FairseqEncoder):
        component_type = "encoder"
    elif isinstance(component, FairseqDecoder):
        component_type = "decoder"
    else:
        raise ValueError(
            "component to load must be either a FairseqEncoder or "
            "FairseqDecoder. Loading other component types are not supported."
        )
    pretrained_output_proj = None
    component_state_dict = OrderedDict()
    for key in state["model"].keys():
        if key.startswith(component_type):
            # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
            component_subkey = key[len(component_type) + 1 :]
            # print(f'component_subkey: {component_subkey}, key: {key}')
            if component_subkey == "embed_tokens.weight":
                # change the shape of the embedding matrix
                # print(state["model"][key].shape)
                # print(embed.weight.shape)
                embed.weight[:state["model"][key].shape[0]] = state["model"][key]
                component_state_dict[component_subkey] = embed.weight
            elif component_subkey == "output_projection.weight":
                if pretrain is not None:
                    # skip it, the decoder4mention will initialize it later
                    continue
                else:
                    pretrained_output_proj = state["model"][key]
            else:
                component_state_dict[component_subkey] = state["model"][key]
            # component_state_dict[component_subkey] = state["model"][key]
    component.load_state_dict(component_state_dict, strict=False)

    return component, embed, pretrained_output_proj
