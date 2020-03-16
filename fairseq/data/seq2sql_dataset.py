# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
import copy
import torch.nn as nn


from . import data_utils, FairseqDataset, dictionary
from collections import Counter
from fairseq import metrics, options, utils

def load_random_embedding(fname):
	fname = open(fname,'r')
	embed_tokens = []
	for line in fname.readlines():
		pieces = line.strip().split(" ")
		embed_tokens.append([float(weight) for weight in pieces])    
	return torch.Tensor(embed_tokens)        


def copy_prev_embedding(embed_path, dictionary, embed_dim, prev_embedded_tokens_path):
	num_embeddings = len(dictionary)
	padding_idx = dictionary.pad()
	embed_tokens = nn.Embedding(num_embeddings, embed_dim, padding_idx)
	prev_embedded_tokens = load_random_embedding(prev_embedded_tokens_path)
	embed_tokens.weight = nn.Parameter(prev_embedded_tokens)
	embed_dict = utils.parse_embedding(embed_path)
	utils.print_embed_overlap(embed_dict, dictionary)
	return utils.load_embedding(embed_dict, dictionary, embed_tokens)



logger = logging.getLogger(__name__)

def get_valid_indices(sequence,mapping_dict,len_sql_dict,unk_idx, src_dict, sql_dict):      
	valid_indices = list(np.arange(len_sql_dict))
	valid_indices.remove(unk_idx)
	for i in set(sequence):
		try:
			valid_indices.append(mapping_dict[i])
		except:
			print(len(sequence))
			s = " ".join(src_dict.symbols[k] for k in sequence)
			print(s)
			print(src_dict.symbols[i])
			
	
	return sorted(valid_indices)

def collate(
	samples, src_embedding, tgt_embedding, src_dict, sql_dict, pad_idx, eos_idx, unk_idx, left_pad_source=False, left_pad_target=False,
	input_feeding=True, eot_symbol=4, mapping_dict=None, len_sql_dict=53
):
	if len(samples) == 0:
		return {}

	def merge(key, left_pad, move_eos_to_beginning=False):
		return data_utils.collate_tokens(
			[s[key] for s in samples],
			pad_idx, eos_idx, left_pad, move_eos_to_beginning,
		)

	id = torch.LongTensor([s['id'] for s in samples])
	src_tokens = merge('source', left_pad=left_pad_source)
	# sort by descending source length
	src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
	src_lengths, sort_order = src_lengths.sort(descending=True)
	id = id.index_select(0, sort_order)
	src_tokens = src_tokens.index_select(0, sort_order)
	flatten_source = [s['source'].flatten().tolist() for s in samples]
	col_lengths_unordered = [s.index(eot_symbol) for s in flatten_source]
	col_lengths = torch.LongTensor(col_lengths_unordered).index_select(0,sort_order)
	
	valid_indices = [get_valid_indices(flatten_source[s][:col_lengths_unordered[s]],mapping_dict,len_sql_dict, unk_idx, src_dict, sql_dict) for s in sort_order.flatten().tolist()]

	prev_output_tokens = None
	target = None
	if samples[0].get('target', None) is not None:
		target = merge('target', left_pad=left_pad_target)
		target = target.index_select(0, sort_order)
		tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
		ntokens = sum(len(s['target']) for s in samples)

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
		ntokens = sum(len(s['source']) for s in samples)
   

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
			'tgt_embedding': tgt_embedding,
			'valid_indices': valid_indices,
		},
		'target': target,
		#'sql_dict': target_dict,
	}
	if prev_output_tokens is not None:
		batch['net_input']['prev_output_tokens'] = prev_output_tokens
		
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
		self, src, src_sizes, src_dict,  
		sql, sql_sizes, sql_dict,  
		encoder_embed_path, encoder_embed_dim,
		decoder_embed_path, decoder_embed_dim,
		encoder_random_embedding_path,
		decoder_random_embedding_path,
		left_pad_source=False, left_pad_target=False,
		max_source_positions=1500, max_target_positions=1024,
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
		self.eot_symbol = self.src_dict.index('<EOT>')
		self.eov_symbol = self.src_dict.index('<EOV>')
		
		#print()
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
		self.mapping_dict = None
		self.create_mapper(src_dict, sql_dict)
		self.src_embedding = copy_prev_embedding(encoder_embed_path, src_dict, encoder_embed_dim, encoder_random_embedding_path)
		self.tgt_embedding = copy_prev_embedding(decoder_embed_path, sql_dict, decoder_embed_dim, decoder_random_embedding_path)

		#print(self.tgt_embedding)
	def __getitem__(self, index):
		sql_item = self.sql[index] 
		src_item = self.src[index]
		
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
			samples, self.src_embedding, self.tgt_embedding, self.src_dict, self.sql_dict,  
			pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(), 
			unk_idx=self.src_dict.unk(),  
			left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
			input_feeding=self.input_feeding, eot_symbol=self.eot_symbol, mapping_dict=self.mapping_dict, 
			len_sql_dict=self.eov_symbol + 1
		)


	def create_mapper(self,src_dict,sql_dict):
		new_dict = {}
		src_tokens = src_dict.symbols
		sql_tokens = sql_dict.symbols
		common_symbols = set(src_tokens).intersection(sql_tokens)
		for c in common_symbols:
			new_dict[src_dict.index(c)] = sql_dict.index(c)
		self.mapping_dict = new_dict

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
