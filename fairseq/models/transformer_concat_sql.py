# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, NamedTuple, Optional

from fairseq.models.fairseq_encoder import EncoderOut
from fairseq import options, utils
from fairseq.models import (
	FairseqEncoder,
	FairseqIncrementalDecoder,
	FairseqEncoderDecoderModel,
	register_model,
	register_model_architecture,
)
from fairseq.modules import (
	AdaptiveSoftmax,
	LayerNorm,
	PositionalEmbedding,
	SinusoidalPositionalEmbedding,
	TransformerDecoderLayer,
	TransformerEncoderLayer,)

from torch import Tensor

DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5


@register_model('transformer-concatseq2seq')
class TransformerConcatSeq2Seq(FairseqEncoderDecoderModel):
	def __init__(self, encoder, decoder):
		super().__init__(encoder, decoder)
		self.args = args
		self.supports_align_args = True


	@staticmethod
	def add_args(parser):
		"""Add model-specific arguments to the parser."""
		# fmt: off
		parser.add_argument('--activation-fn',
							choices=utils.get_available_activation_fns(),
							help='activation function to use')
		parser.add_argument('--dropout', type=float, metavar='D',
							help='dropout probability')
		parser.add_argument('--attention-dropout', type=float, metavar='D',
							help='dropout probability for attention weights')
		parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
							help='dropout probability after activation in FFN.')
		parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
							help='encoder embedding dimension')
		parser.add_argument('--word-encoder-embed-dim', type=int, metavar='N',
							help='word embedding dimension')

		parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
							help='path to pre-trained encoder embedding')
		parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
							help='encoder embedding dimension for FFN')
		parser.add_argument('--encoder-layers', type=int, metavar='N',
							help='num encoder layers')
		parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
							help='num encoder attention heads')
		parser.add_argument('--encoder-normalize-before', action='store_true',
							help='apply layernorm before each encoder block')
		parser.add_argument('--encoder-learned-pos', action='store_true',
							help='use learned positional embeddings in the encoder')
		parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
							help='decoder embedding dimension')
		parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
							help='path to pre-trained decoder embedding')

		parser.add_argument('--encoder-freeze-embed', action='store_true',
							help='freeze encoder embeddings')

		parser.add_argument('--decoder-freeze-embed', action='store_true',
							help='freeze decoder embeddings')
		parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
							help='decoder hidden size')
		parser.add_argument('--decoder-layers', type=int, metavar='N',
							help='number of decoder layers')
		parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
							help='decoder output embedding dimension')
		parser.add_argument('--decoder-attention', type=str, metavar='BOOL',
							help='decoder attention')
		parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
							help='comma separated list of adaptive softmax cutoff points. '
								 'Must be used with adaptive_loss criterion')
		parser.add_argument('--share-decoder-input-output-embed', default=False,
							action='store_true',
							help='share decoder input and output embeddings')
		parser.add_argument('--share-all-embeddings', default=False, action='store_true',
							help='share encoder, decoder and output embeddings'
								 ' (requires shared dictionary and embed dim)')
		parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
						   help='if set, disables positional embeddings (outside self attention)')
	   
		parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
							help='sets adaptive softmax dropout for the tail projections')


		parser.add_argument('--no-cross-attention', default=False, action='store_true',
							help='do not perform cross-attention')
		parser.add_argument('--cross-self-attention', default=False, action='store_true',
							help='perform cross+self-attention')
		parser.add_argument('--layer-wise-attention', default=False, action='store_true',
							help='perform layer-wise attention (cross-attention or cross+self-attention)')
		# args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
		parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
							help='LayerDrop probability for encoder')
		parser.add_argument('--encoder-layers-to-keep', default=None,
							help='which layers to *keep* when pruning as a comma-separated list')
 
		parser.add_argument('--layernorm-embedding', action='store_true',
							help='add layernorm to embedding')
		parser.add_argument('--no-scale-embedding', action='store_true',
							help='if True, dont scale embeddings')


		# Granular dropout settings (if not specified these default to --dropout)
		parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
							help='dropout probability for encoder input embedding')
		parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
							help='dropout probability for encoder output')
		parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
							help='dropout probability for decoder input embedding')
		parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
							help='dropout probability for decoder output')
		# fmt: on

	@classmethod
	def build_model(cls, args, task):
		"""Build a new model instance."""
		# make sure that all args are properly defaulted (in case there are any new ones)
		base_architecture(args)
		if args.encoder_layers_to_keep:
			args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

	
		max_source_positions = getattr(args, 'max_source_positions', DEFAULT_MAX_SOURCE_POSITIONS)
		max_target_positions = getattr(args, 'max_target_positions', DEFAULT_MAX_TARGET_POSITIONS)

		def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
			num_embeddings = len(dictionary)
			padding_idx = dictionary.pad()
			embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
			embed_dict = utils.parse_embedding(embed_path)
			utils.print_embed_overlap(embed_dict, dictionary)
			return utils.load_embedding(embed_dict, dictionary, embed_tokens)

		if args.encoder_embed_path:
			pretrained_encoder_embed = load_pretrained_embedding_from_file(
				args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
		else:
			num_embeddings = len(task.source_dictionary)
			pretrained_encoder_embed = Embedding(
				num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
			)

		if args.share_all_embeddings:
			# double check all parameters combinations are valid
			if task.source_dictionary != task.target_dictionary:
				raise ValueError('--share-all-embeddings requires a joint dictionary')
			if args.decoder_embed_path and (
					args.decoder_embed_path != args.encoder_embed_path):
				raise ValueError(
					'--share-all-embed not compatible with --decoder-embed-path'
				)
			if args.encoder_embed_dim != args.decoder_embed_dim:
				raise ValueError(
					'--share-all-embeddings requires --encoder-embed-dim to '
					'match --decoder-embed-dim'
				)
			pretrained_decoder_embed = pretrained_encoder_embed
			args.share_decoder_input_output_embed = True
		else:
			# separate decoder input embeddings
			pretrained_decoder_embed = None
			if args.decoder_embed_path:
				pretrained_decoder_embed = load_pretrained_embedding_from_file(
					args.decoder_embed_path,
					task.target_dictionary,
					args.decoder_embed_dim
				)
		# one last double check of parameter combinations
		if args.share_decoder_input_output_embed and (
				args.decoder_embed_dim != args.decoder_out_embed_dim):
			raise ValueError(
				'--share-decoder-input-output-embeddings requires '
				'--decoder-embed-dim to match --decoder-out-embed-dim'
			)

		if args.encoder_freeze_embed:
			pretrained_encoder_embed.weight.requires_grad = False
		if args.decoder_freeze_embed:
			pretrained_decoder_embed.weight.requires_grad = False

		encoder = TransformerEncoder(args, task.source_dictionary, args.word_encoder_embed_dim, args.encoder_embed_dim)
		decoder = LSTMDecoder(
			dictionary=task.target_dictionary,
			embed_dim=args.decoder_embed_dim,
			hidden_size=args.decoder_hidden_size,
			out_embed_dim=args.decoder_out_embed_dim,
			num_layers=args.decoder_layers,
			dropout_in=args.decoder_dropout_in,
			dropout_out=args.decoder_dropout_out,
			attention=options.eval_bool(args.decoder_attention),
			encoder_output_units=encoder.output_units,
			pretrained_embed=pretrained_decoder_embed,
			share_input_output_embed=args.share_decoder_input_output_embed,
			adaptive_softmax_cutoff=(
				options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
				if args.criterion == 'adaptive_loss' else None
			),
			max_target_positions=max_target_positions
		)
		return cls(encoder, decoder)

	def forward(self, src_tokens, src_lengths, src_embedding, tgt_embedding, valid_indices, prev_output_tokens, **kwargs):
		encoder_out = self.encoder(src_tokens, src_lengths, src_embedding)
		decoder_out = self.decoder(prev_output_tokens, valid_indices, tgt_embedding, encoder_out=encoder_out, **kwargs)
		return decoder_out

	def forward_decoder(self, prev_output_tokens, valid_indices, tgt_embedding, encoder_out=None, incremental_state=None, **kwargs):
		return self.decoder(prev_output_tokens, valid_indices, tgt_embedding, encoder_out=encoder_out, incremental_state=None, **kwargs)



class TransformerSQLEncoder(FairseqEncoder):
	"""
	Transformer encoder consisting of *args.encoder_layers* layers. Each layer
	is a :class:`TransformerEncoderLayer`.
	Args:
		args (argparse.Namespace): parsed command-line arguments
		dictionary (~fairseq.data.Dictionary): encoding dictionary
		embed_tokens (torch.nn.Embedding): input embedding
	"""

	def __init__(self, args, dictionary, word_encoder_embed_dim, encoder_embed_dim):
		super().__init__(dictionary)
		self.register_buffer("version", torch.Tensor([3]))

		self.dropout = args.dropout
		self.encoder_layerdrop = args.encoder_layerdrop

		if word_encoder_embed_dim != encoder_embed_dim:
			self.word_projection_layer = Linear(word_encoder_embed_dim, encoder_embed_dim)
		else:
			self.word_projection_layer = None
		embed_dim = encoder_embed_dim
		self.output_units = encoder_embed_dim
		self.padding_idx = embed_tokens.padding_idx
		self.max_source_positions = args.max_source_positions


		self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

		self.embed_positions = (
			PositionalEmbedding(
				args.max_source_positions,
				embed_dim,
				self.padding_idx,
				learned=args.encoder_learned_pos,
			)
			if not args.no_token_positional_embeddings
			else None
		)

		self.layer_wise_attention = getattr(args, "layer_wise_attention", False)

		self.layers = nn.ModuleList([])
		self.layers.extend(
			[TransformerEncoderLayer(args) for i in range(args.encoder_layers)]
		)
		self.num_layers = len(self.layers)

		if args.encoder_normalize_before:
			self.layer_norm = LayerNorm(embed_dim)
		else:
			self.layer_norm = None
		if getattr(args, "layernorm_embedding", False):
			self.layernorm_embedding = LayerNorm(embed_dim)
		else:
			self.layernorm_embedding = None

	def forward_embedding(self, src_tokens, src_embedding):
		# embed tokens and positions
		x = embed = self.embed_scale * src_embedding(src_tokens)
		if self.word_projection_layer is not None:
			x = embed = self.word_projection_layer(embed)
		if self.embed_positions is not None:
			x = embed + self.embed_positions(src_tokens)
		if self.layernorm_embedding is not None:
			x = self.layernorm_embedding(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		return x, embed

	def forward(
		self,
		src_tokens,
		src_lengths,
		src_embedding,
		cls_input: Optional[Tensor] = None,
		return_all_hiddens: bool = False,
	):
		"""
		Args:
			src_tokens (LongTensor): tokens in the source language of shape
				`(batch, src_len)`
			src_lengths (torch.LongTensor): lengths of each source sentence of
				shape `(batch)`
			return_all_hiddens (bool, optional): also return all of the
				intermediate hidden states (default: False).
		Returns:
			namedtuple:
				- **encoder_out** (Tensor): the last encoder layer's output of
				  shape `(src_len, batch, embed_dim)`
				- **encoder_padding_mask** (ByteTensor): the positions of
				  padding elements of shape `(batch, src_len)`
				- **encoder_embedding** (Tensor): the (scaled) embedding lookup
				  of shape `(batch, src_len, embed_dim)`
				- **encoder_states** (List[Tensor]): all intermediate
				  hidden states of shape `(src_len, batch, embed_dim)`.
				  Only populated if *return_all_hiddens* is True.
		"""
		if self.layer_wise_attention:
			return_all_hiddens = True

		x, encoder_embedding = self.forward_embedding(src_tokens, src_embedding)

		# B x T x C -> T x B x C
		x = x.transpose(0, 1)

		# compute padding mask
		encoder_padding_mask = src_tokens.eq(self.padding_idx)

		encoder_states = [] if return_all_hiddens else None

		# encoder layers
		for layer in self.layers:
			# add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
			dropout_probability = torch.empty(1).uniform_()
			if not self.training or (dropout_probability > self.encoder_layerdrop):
				x = layer(x, encoder_padding_mask)
				if return_all_hiddens:
					assert encoder_states is not None
					encoder_states.append(x)

		if self.layer_norm is not None:
			x = self.layer_norm(x)
			if return_all_hiddens:
				encoder_states[-1] = x

		return EncoderOut(
			encoder_out=x,  # T x B x C
			encoder_padding_mask=encoder_padding_mask,  # B x T
			encoder_embedding=encoder_embedding,  # B x T x C
			encoder_states=encoder_states,  # List[T x B x C]
		)

	@torch.jit.export
	def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
		"""
		Reorder encoder output according to *new_order*.
		Args:
			encoder_out: output from the ``forward()`` method
			new_order (LongTensor): desired order
		Returns:
			*encoder_out* rearranged according to *new_order*
		"""
		new_encoder_out: Dict[str, Tensor] = {}

		new_encoder_out["encoder_out"] = (
			encoder_out.encoder_out
			if encoder_out.encoder_out is None
			else encoder_out.encoder_out.index_select(1, new_order)
		)
		new_encoder_out["encoder_padding_mask"] = (
			encoder_out.encoder_padding_mask
			if encoder_out.encoder_padding_mask is None
			else encoder_out.encoder_padding_mask.index_select(0, new_order)
		)
		new_encoder_out["encoder_embedding"] = (
			encoder_out.encoder_embedding
			if encoder_out.encoder_embedding is None
			else encoder_out.encoder_embedding.index_select(0, new_order)
		)

		encoder_states = encoder_out.encoder_states
		if encoder_states is not None:
			for idx, state in enumerate(encoder_states):
				encoder_states[idx] = state.index_select(1, new_order)

		return EncoderOut(
			encoder_out=new_encoder_out["encoder_out"],  # T x B x C
			encoder_padding_mask=new_encoder_out["encoder_padding_mask"],  # B x T
			encoder_embedding=new_encoder_out["encoder_embedding"],  # B x T x C
			encoder_states=encoder_states,  # List[T x B x C]
		)

	def max_positions(self):
		"""Maximum input length supported by the encoder."""
		if self.embed_positions is None:
			return self.max_source_positions
		return min(self.max_source_positions, self.embed_positions.max_positions)

	def buffered_future_mask(self, tensor):
		dim = tensor.size(0)
		if (
			not hasattr(self, "_future_mask")
			or self._future_mask is None
			or self._future_mask.device != tensor.device
		):
			self._future_mask = torch.triu(
				utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
			)
			if self._future_mask.size(0) < dim:
				self._future_mask = torch.triu(
					utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
				)
		return self._future_mask[:dim, :dim]

	def upgrade_state_dict_named(self, state_dict, name):
		"""Upgrade a (possibly old) state dict for new versions of fairseq."""
		if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
			weights_key = "{}.embed_positions.weights".format(name)
			if weights_key in state_dict:
				print("deleting {0}".format(weights_key))
				del state_dict[weights_key]
			state_dict[
				"{}.embed_positions._float_tensor".format(name)
			] = torch.FloatTensor(1)
		for i in range(self.num_layers):
			# update layer norms
			self.layers[i].upgrade_state_dict_named(
				state_dict, "{}.layers.{}".format(name, i)
			)

		version_key = "{}.version".format(name)
		if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
			# earlier checkpoints did not normalize after the stack of layers
			self.layer_norm = None
			self.normalize = False
			state_dict[version_key] = torch.Tensor([1])
		return state_dict








class AttentionLayer(nn.Module):
	def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
		super().__init__()

		self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
		self.output_proj = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

	def forward(self, input, source_hids, encoder_padding_mask):
		# input: bsz x input_embed_dim
		# source_hids: srclen x bsz x source_embed_dim

		# x: bsz x source_embed_dim
		x = self.input_proj(input)

		# compute attention
		attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

		# don't attend over padding
		if encoder_padding_mask is not None:
			attn_scores = attn_scores.float().masked_fill_(
				encoder_padding_mask,
				float('-inf')
			).type_as(attn_scores)  # FP16 support: cast to float and back

		attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

		# sum weighted sources
		x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

		x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
		return x, attn_scores


class LSTMDecoder(FairseqIncrementalDecoder):
	"""LSTM decoder."""
	def __init__(
		self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
		num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
		encoder_output_units=512, pretrained_embed=None,
		share_input_output_embed=False, adaptive_softmax_cutoff=None,
		max_target_positions=DEFAULT_MAX_TARGET_POSITIONS
	):
		super().__init__(dictionary)
		self.dropout_in = dropout_in
		self.dropout_out = dropout_out
		self.hidden_size = hidden_size
		self.share_input_output_embed = share_input_output_embed
		self.need_attn = True
		self.max_target_positions = max_target_positions
		
		self.adaptive_softmax = None
		self.num_embeddings = len(dictionary)
		padding_idx = dictionary.pad()
		if pretrained_embed is None:
			self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
		else:
			self.embed_tokens = pretrained_embed

		self.encoder_output_units = encoder_output_units

		# disable input feeding if there is no encoder
		# input feeding is described in arxiv.org/abs/1508.04025
		input_feed_size = 0 if encoder_output_units == 0 else hidden_size
		self.layers = nn.ModuleList([
			LSTMCell(
				input_size=input_feed_size + embed_dim if layer == 0 else hidden_size,
				hidden_size=hidden_size,
			)
			for layer in range(num_layers)
		])
		if attention:
			# TODO make bias configurable
			self.attention = AttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
		else:
			self.attention = None
		if hidden_size != out_embed_dim:
			self.additional_fc = Linear(hidden_size, out_embed_dim)
		if adaptive_softmax_cutoff is not None:
			# setting adaptive_softmax dropout to dropout_out for now but can be redefined
			self.adaptive_softmax = AdaptiveSoftmax(self.num_embeddings, hidden_size, adaptive_softmax_cutoff,
													dropout=dropout_out)
		elif not self.share_input_output_embed:
			self.fc_out = Linear(out_embed_dim, self.num_embeddings, dropout=dropout_out)

	def forward(self, prev_output_tokens, valid_indices, tgt_embedding, encoder_out=None, incremental_state=None, **kwargs):
		x, attn_scores = self.extract_features(
			prev_output_tokens, valid_indices, tgt_embedding, encoder_out, incremental_state
		)
		return self.output_layer(x,tgt_embedding,valid_indices), attn_scores

	def extract_features(
		self, prev_output_tokens, valid_indices, tgt_embedding,  encoder_out, incremental_state=None
	):
		"""
		Similar to *forward* but only return features.
		"""
		if encoder_out is not None:
			encoder_padding_mask = encoder_out['encoder_padding_mask']
			encoder_out = encoder_out['encoder_out']
		else:
			encoder_padding_mask = None
			encoder_out = None

		if incremental_state is not None:
			prev_output_tokens = prev_output_tokens[:, -1:]
		bsz, seqlen = prev_output_tokens.size()

		# get outputs from encoder
		if encoder_out is not None:
			encoder_outs = encoder_out
			srclen = encoder_outs.size(0)
		else:
			srclen = None

		# embed tokens
		if tgt_embedding is not None:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			tgt_embedding.to(device)
			#print(prev_output_tokens)
			x = tgt_embedding(prev_output_tokens.cuda())
		else:   
			x = self.embed_tokens(prev_output_tokens)

#        x = self.embed_tokens(prev_output_tokens)
		x = F.dropout(x, p=self.dropout_in, training=self.training)

		# B x T x C -> T x B x C
		x = x.transpose(0, 1)

		# initialize previous states (or get from cache during incremental generation)
		cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
		if cached_state is not None:
			prev_hiddens, prev_cells, input_feed = cached_state
		else:
			# setup zero cells, since there is no encoder
			num_layers = len(self.layers)
			zero_state = x.new_zeros(bsz, self.hidden_size)
			prev_hiddens = [zero_state for i in range(num_layers)]
			prev_cells = [zero_state for i in range(num_layers)]
			input_feed = None

		assert srclen is not None or self.attention is None, \
			"attention is not supported if there are no encoder outputs"
		attn_scores = x.new_zeros(srclen, seqlen, bsz) if self.attention is not None else None
		outs = []
		for j in range(seqlen):
			# input feeding: concatenate context vector from previous time step
			if input_feed is not None:
				input = torch.cat((x[j, :, :], input_feed), dim=1)
			else:
				input = x[j]

			for i, rnn in enumerate(self.layers):
				# recurrent cell
				hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

				# hidden state becomes the input to the next layer
				input = F.dropout(hidden, p=self.dropout_out, training=self.training)

				# save state for next time step
				prev_hiddens[i] = hidden
				prev_cells[i] = cell

			# apply attention using the last layer's hidden state
			if self.attention is not None:
				out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
			else:
				out = hidden
			out = F.dropout(out, p=self.dropout_out, training=self.training)

			# input feeding
			if input_feed is not None:
				input_feed = out

			# save final output
			outs.append(out)

		# cache previous states (no-op except during incremental generation)
		utils.set_incremental_state(
			self, incremental_state, 'cached_state',
			(prev_hiddens, prev_cells, input_feed),
		)

		# collect outputs across time steps
		x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

		# T x B x C -> B x T x C
		x = x.transpose(1, 0)

		if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
			x = self.additional_fc(x)
			x = F.dropout(x, p=self.dropout_out, training=self.training)

		# srclen x tgtlen x bsz -> bsz x tgtlen x srclen
		if not self.training and self.need_attn and self.attention is not None:
			attn_scores = attn_scores.transpose(0, 2)
		else:
			attn_scores = None
		return x, attn_scores



	def get_decoder_padding_mask(self, valid_indices):
		decoder_padding_mask = torch.zeros(len(valid_indices), self.num_embeddings)
		for c in range(len(valid_indices)):
			try:
				decoder_padding_mask[c,valid_indices[c]] = 1.0
			except:
				print(c)
		decoder_padding_mask = (decoder_padding_mask != 1) #Elements that need to be masked should be True
		return decoder_padding_mask
	

	def output_layer(self, x, tgt_embedding, valid_indices=None):
		"""Project features to the vocabulary size."""
		x = F.linear(x, tgt_embedding.weight)
		if valid_indices!=None:
			decoder_padding_mask = self.get_decoder_padding_mask(valid_indices)
			decoder_padding_mask = decoder_padding_mask.repeat(1,x.size()[1]).view(x.size())
		x = x.float().masked_fill_(decoder_padding_mask.cuda(),float('-1e-32')).type_as(x)
		return x	

	def reorder_incremental_state(self, incremental_state, new_order):
		super().reorder_incremental_state(incremental_state, new_order)
		cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
		if cached_state is None:
			return

		def reorder_state(state):
			if isinstance(state, list):
				return [reorder_state(state_i) for state_i in state]
			elif state is not None:
				return state.index_select(0, new_order)
			else:
				return None

		new_state = tuple(map(reorder_state, cached_state))
		utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

	def max_positions(self):
		"""Maximum output length supported by the decoder."""
		return self.max_target_positions

	def make_generation_fast_(self, need_attn=False, **kwargs):
		self.need_attn = need_attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
	m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
	nn.init.uniform_(m.weight, -0.1, 0.1)
	nn.init.constant_(m.weight[padding_idx], 0)
	return m


def LSTM(input_size, hidden_size, **kwargs):
	m = nn.LSTM(input_size, hidden_size, **kwargs)
	for name, param in m.named_parameters():
		if 'weight' in name or 'bias' in name:
			param.data.uniform_(-0.1, 0.1)
	return m


def LSTMCell(input_size, hidden_size, **kwargs):
	m = nn.LSTMCell(input_size, hidden_size, **kwargs)
	for name, param in m.named_parameters():
		if 'weight' in name or 'bias' in name:
			param.data.uniform_(-0.1, 0.1)
	return m


def Linear(in_features, out_features, bias=True, dropout=0):
	"""Linear layer (input: N x T x C)"""
	m = nn.Linear(in_features, out_features, bias=bias)
	m.weight.data.uniform_(-0.1, 0.1)
	if bias:
		m.bias.data.uniform_(-0.1, 0.1)
	return m



@register_model_architecture('transformer-concatseq2seq', 'transformer-concatseq2seq')
def base_architecture(args):
	args.dropout = getattr(args, 'dropout', 0.1)
	args.encoder_embed_path = getattr(args, "encoder_embed_path", args.encoder_embed_path)
	args.word_encoder_embed_dim = getattr(args, "word_encoder_embed_dim", 300)
	args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
	args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
	args.encoder_layers = getattr(args, "encoder_layers", 6)
	args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
	args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
	args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
	args.encoder_freeze_embed = getattr(args, 'encoder_freeze_embed',True)

	args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.decoder_embed_dim)
	args.decoder_embed_path = getattr(args, 'decoder_embed_path', args.decoder_embed_path)
	args.decoder_freeze_embed = getattr(args, 'decoder_freeze_embed', True)
	args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
	args.decoder_layers = getattr(args, 'decoder_layers', 2)
	args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim',args.decoder_out_embed_dim)
	args.decoder_attention = getattr(args, 'decoder_attention', '1')
	args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
	args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
	args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
	args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
	args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '10000,50000,200000')
	args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)

	args.left_pad_source = getattr(args, 'left_pad_source', False)
	args.left_pad_target = getattr(args, 'left_pad_target', False)

	args.attention_dropout = getattr(args, "attention_dropout", 0.3)
	args.activation_dropout = getattr(args, "activation_dropout", 0.3)
	args.activation_fn = getattr(args, "activation_fn", "relu")
	args.adaptive_input = getattr(args, "adaptive_input", False)
	args.no_cross_attention = getattr(args, "no_cross_attention", False)
	args.cross_self_attention = getattr(args, "cross_self_attention", False)
	args.layer_wise_attention = getattr(args, "layer_wise_attention", False)
	args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
	args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
	args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)