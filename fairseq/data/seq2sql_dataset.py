# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
import copy

from . import data_utils, FairseqDataset, dictionary
from collections import Counter
from fairseq import metrics, options, utils
from fairseq.models.lstm import Embedding




logger = logging.getLogger(__name__)
def convert_to_dict_object(dict_name):
    f = open('temp_dict.txt','w')
    for k,v in dict_name.items():
        f.write(k + ' ' + str(v) + '\n')
    f.close()

    return dictionary.Dictionary.load('temp_dict.txt')


class SrcObject(object):
    """docstring for SrcObject"""
    def __init__(self, id_, sequence, basic_dict, col_length, col_dict=None):
        super(SrcObject, self).__init__()
        self.id = id_
        #print(basic_dict.indices)
        self.sequence = [basic_dict.symbols[int(s)] for s in sequence] #optimize this!
        
        self.basic_dict = copy.deepcopy(basic_dict)
        self.col_length = col_length
        self.col_dict = None
        self.index_sequence = None
        self.set_col_dict()
        self.convert_sequence_to_ids()

    def set_col_dict(self):
        col_set = Counter(self.sequence[:self.col_length])
        col_set = convert_to_dict_object(col_set)
        self.basic_dict.update(col_set)
        self.col_dict = col_set

    def convert_sequence_to_ids(self):
        self.index_sequence = torch.LongTensor([self.basic_dict.index(s) for s in self.sequence])
         

class SqlObject(object):
    """docstring for SqlObject"""
    def __init__(self, id_, sequence, basic_dict, col_dict):
        super(SqlObject, self).__init__()
        self.id = id_
        self.sequence = sequence
        self.sequence = [basic_dict.symbols[int(s)] for s in sequence]
        self.basic_dict = basic_dict
        #self.col_dict = convert_to_dict_object(col_dict)
        self.basic_dict.update(col_dict)
        self.convert_sequence_to_ids()
       

    def convert_sequence_to_ids(self):
        self.index_sequence = torch.LongTensor([self.basic_dict.index(s) for s in self.sequence])


def collate(
    samples, pad_idx, eos_idx, src_embedding, tgt_embedding,  left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key].index_sequence for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )


    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    #src_lengths = torch.LongTensor([s['source'].index_sequence.numel() for s in samples])
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    src_dict = [samples[k]['source'].basic_dict for k in sort_order]
    col_lengths = torch.LongTensor([s['column_sizes'] for s in samples]).index_select(0,sort_order)
    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        target_dict = [samples[k]['target'].basic_dict for k in sort_order]
        sql_lengths = torch.LongTensor([s['target'].index_sequence.numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target'].sequence) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source'].sequence) for s in samples)

     

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            #'src_dict': src_dict, 
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'col_lengths': col_lengths,
            'src_embedding': src_embedding,
        },
        'target': target,
        #'sql_dict': target_dict,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
        batch['net_input']['tgt_embedding'] = tgt_embedding #where will this go?

    return batch


class Seq2SqlPairDataSet(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        src_column_sizes (List[int]): column lengths 
        sql (torch.utils.data.Dataset, optional): target dataset to wrap
        sql_sizes (List[int], optional): target sentence lengths
        sql_dict (~fairseq.data.Dictionary): sql vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
    """

    def __init__(
        self, src, src_sizes, src_dict, col_sizes,  
        sql, sql_sizes, sql_dict,  embed_path, embed_dim, 
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        append_bos = False
    ):
        if sql_dict is not None:
            assert src_dict.pad() == sql_dict.pad()
            assert src_dict.eos() == sql_dict.eos()
            assert src_dict.unk() == sql_dict.unk()
        self.src = src
        self.sql = sql
        self.src_dict = src_dict
        self.sql_dict = sql_dict
        self.src_sizes = np.array(src_sizes)
        self.col_sizes = np.array(col_sizes)
        self.sql_sizes = np.array(sql_sizes) 
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.append_bos = append_bos

        self.src_embedding = load_pretrained_embedding_from_file(
                        embed_path, self.src_dict, embed_dim)

        self.tgt_embedding = load_pretrained_embedding_from_file(
                        embed_path, self.sql_dict, embed_dim)


    def __getitem__(self, index):
        sql_item = self.sql[index] 
        src_item = self.src[index]
        col_item = self.col_sizes[index]
        # Append EOS to end of sql sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use sql_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.sql_dict.eos() if self.sql_dict else self.src_dict.eos()
            if self.sql and self.sql[index][-1] != eos:
                sql_item = torch.cat([self.sql[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.sql_dict.bos() if self.sql_dict else self.src_dict.bos()
            if self.sql and self.sql[index][0] != bos:
                sql_item = torch.cat([torch.LongTensor([bos]), self.sql[index]])

            bos = self.src_dict.bos()
            if self.src[index][-1] != bos:
                src_item = torch.cat([torch.LongTensor([bos]), self.src[index]])
                col_item = col_item + 1
        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]
        #src_item = SrcObject(index, src_item, self.src_dict, col_item)
        #sql_item = SqlObject(index, sql_item, self.sql_dict, src_item.col_dict)
        example = {
            'id': index,
            'source': src_item,
            'target': sql_item,
            'column_sizes': col_item,
        }
        return example

    def __len__(self):
        return len(self.src)

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
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, sql_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, sql_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(), self.src_embedding, self.tgt_embedding, 
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.sql_sizes[index])

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.sql_sizes[index])

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.sql_sizes is not None:
            indices = indices[np.argsort(self.sql_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.sql, 'supports_prefetch', False))
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.sql is not None:
            self.sql.prefetch(indices)
