from typing import List
import torch

from fairseq.tokenizer import tokenize_line
from fairseq.data import Dictionary, FairseqDataset, LanguagePairDataset, data_utils
from collections import defaultdict

from .entity_dictionary import LangWithEntityDictionary
from .utils import combine_ne_with_text, tag_entity

import random

"""
For the training time, we need to consider two things
1. What to put in 'net_input'
    Method 1, adjacent tokens in x are replaced by placeholder.
    Method 2, raw x.
    Method 3, raw x.
2. What other things need to compute the loss.
    Method 1, adjacent tokens in y are replaced by placeholder.
    Method 2, x entity sequence, y entity sequence, y tokens
    Method 3, x entity sequence, y entity sequence (adjacent tokens combined), BERT processed entity (should be provided by task, not here).
"""


class EntityTranslationDataset(FairseqDataset):
    def __init__(self,
                 lang_pair: LanguagePairDataset,
                 ne_pair: LanguagePairDataset,
                 mode: int,
                 src_dict: LangWithEntityDictionary,
                 tgt_dict: LangWithEntityDictionary,
                 ignore_entity_type: List[str],
                 ne_drop_rate: float,
                 is_train: bool):
        self.lang_pair = lang_pair
        self.ne_pair = ne_pair
        self.mode = mode
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        assert len(self.lang_pair) == len(self.ne_pair), f'Language data lenth {len(self.lang_pair)} != NE data length {len(self.ne_pair)}'

        self.collater_methods = [
            self.collater_mode_0,
            self.collater_mode_1,
            self.collater_mode_2,
            self.collater_mode_3
        ]

        assert 0 <= self.mode < len(self.collater_methods), f'Invalid mode {self.mode}'
        # self.o_index = self.tgt_dict.ne_dict.index('NONE')
        self.ne_drop_rate = ne_drop_rate
        assert 0 <= self.ne_drop_rate <= 1, f'{self.ne_drop_rate}'

        self.is_train = is_train

    # def _drop_entity(self, tensor, eos):
    #     no_entity = torch.ones_like(tensor, dtype=tensor.dtype, device=tensor.device) * self.o_index
    #     mask = tensor != eos
    #     return torch.where(mask, no_entity, tensor)

    def __getitem__(self, index):
        """
        {   id: index,
            lang_pair:
            {
                'id': index,
                'source': source,
                'target': target,
            },
            ne_pair:
            {
                'id': index,
                'source': source,
                'target': target,
            }
        }
        """
        ne_pair = self.ne_pair[index]
        entity_sent_mask = True
        #
        # if self.is_train:
        #     entity_sent_mask = random.random() >= self.ne_drop_rate
        #
        #     if not entity_sent_mask: # Not a entity sentence, only keep 'O' during the training
        #         ne_pair['source'] = self._drop_entity(ne_pair['source'], self.src_dict.eos())
        #         ne_pair['target'] = self._drop_entity(ne_pair['target'], self.tgt_dict.eos())

        return {
            'id': index,
            'lang_pair': self.lang_pair[index],
            'ne_pair': ne_pair,
            'entity_sent_mask': entity_sent_mask
        }

    def __len__(self):
        return len(self.lang_pair)

    def collater_mode_0(self, samples):
        """
        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
            'target': target,

            # below are custom items
            'src_ne_pos': [None, Slice, xxx],
            'origin_src': '' # the text before merge ne
            'origin_tgt': ''
        }
        """
        combined_lang_pair = []

        origin_sent_map = {
            'source': {},
            'target': {}
        }
        for sample in samples:
            sample_id = sample['id']
            combined_src_tokens, _, src_alignment = combine_ne_with_text(sample['lang_pair']['source'], sample['ne_pair']['source'], self.src_dict, self.max_ne_id)
            combined_tgt_tokens, _, _ = combine_ne_with_text(sample['lang_pair']['target'], sample['ne_pair']['target'], self.tgt_dict, self.max_ne_id)
            combined_lang_pair.append({
                'id': sample_id,
                'source': torch.stack(combined_src_tokens),
                'target': torch.stack(combined_tgt_tokens)
            })

            origin_sent_map['source'][sample_id] = sample['lang_pair']['source']
            origin_sent_map['target'][sample_id] = sample['lang_pair']['target']

        batch = self.lang_pair.collater(combined_lang_pair)
        batch_ids = batch['id'].tolist()

        batch['origin_src'] = [origin_sent_map['source'][x] for x in batch_ids]
        batch['origin_tgt'] = [origin_sent_map['target'][x] for x in batch_ids]

        return batch

    def collater_mode_1(self, samples):
        """
        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
            'target': target,

            'ne_pair': ne_pair,
            'ne_source': [],
            'ne_target': []
        }
        """
        batch = self.lang_pair.collater([sample['lang_pair'] for sample in samples])

        id2ne = {sample['id']: sample['ne_pair'] for sample in samples}
        batch_ids = batch['id'].tolist()

        batch['ne_pair'] = [id2ne[x] for x in batch_ids]
        batch['ne_source'] = data_utils.collate_tokens(
            [s['source'] for s in batch['ne_pair']],
            self.ne_pair.src_dict.pad(), self.ne_pair.src_dict.eos(), self.ne_pair.left_pad_source
        )
        batch['ne_target'] = data_utils.collate_tokens(
            [s['target'] for s in batch['ne_pair']],
            self.ne_pair.tgt_dict.pad(), self.ne_pair.tgt_dict.eos(), self.ne_pair.left_pad_target
        )
        return batch

    def collater_mode_2(self, samples):
        return self.collater_mode_1(samples)
    
    def collater_mode_3(self, samples):
        return self.collater_mode_1(samples)

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        return self.collater_methods[self.mode](samples)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.lang_pair.num_tokens(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        lang_pair_size = self.lang_pair.size(index)
        # The NE should have same size as src
        return (lang_pair_size[0], lang_pair_size[1], lang_pair_size[0])

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return self.lang_pair.ordered_indices()

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return self.lang_pair.supports_prefetch and self.ne_pair.supports_prefetch

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        self.lang_pair.prefetch(indices)
        self.ne_pair.prefetch(indices)
