#Based on https://github.com/pytorch/fairseq/tree/master/fairseq
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#task file

from argparse import Namespace
import json
import itertools
import logging
import os

import numpy as np

from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    Seq2SqlPairDataSet,
)

from fairseq.tasks import FairseqTask, register_task

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)

def get_col_sizes(filename):
    f = open(filename)
    size_list = []
    for i in f.readlines():
        size_list.append(int(i.strip()))
    return size_list

def load_seq_sql_dataset(data_path, split, src, src_dict, sql, sql_dict,  
        dataset_impl, upsample_primary, 
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        truncate_source,
        prepend_bos):

    
    src_datasets = []
    sql_datasets = []

    prefix = os.path.join(data_path, split)
    

    src_dataset = data_utils.load_indexed_dataset(prefix + '.' + src, src_dict, dataset_impl)
    col_sizes = get_col_sizes(prefix + '.col')
    if truncate_source:
        src_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDataset(src_dataset, src_dict.eos()),
                max_source_positions - 1,
            ),
            src_dict.eos(),
        )
    src_datasets.append(src_dataset)
    sql_datasets.append(data_utils.load_indexed_dataset(prefix +  '.' + sql, sql_dict, dataset_impl))



    assert len(src_datasets) == len(sql_datasets)

    if len(src_datasets) == 1:
        src_dataset, sql_dataset = src_datasets[0], sql_datasets[0]
    else: #not implemented
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        sql_dataset = ConcatDataset(sql_datasets, sample_ratios)

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(sql_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        sql_dataset = PrependTokenDataset(sql_dataset, sql_dict.bos())

  

    return Seq2SqlPairDataSet(
        src_dataset, src_dataset.sizes, src_dict, col_sizes,
        sql_dataset, sql_dataset.sizes, sql_dict, 
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
    )


@register_task('Seq2Sql')
class Seq2SqlTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source 
        sql_dict (~fairseq.data.Dictionary): dictionary for the target 
        col_dict (~fairseq.data.Dictionary): dictionary for the col 

       .. note::

    
    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')

        parser.add_argument('--left-pad-source', default='False', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')

        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--combine', default=False)
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--add-bos-token', action='store_true',
                            help='prepend beginning of sentence token (<s>)')
    
    def __init__(self, args, src_dict, sql_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.sql_dict = sql_dict



    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        paths = args.data.split(os.pathsep)
        assert len(paths) > 0

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.src.txt'))
        sql_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.sql.txt'))
        
        assert src_dict.pad() == sql_dict.pad()
        assert src_dict.eos() == sql_dict.eos()
        assert src_dict.unk() == sql_dict.unk()
        logger.info('["src"] dictionary: {} types'.format(len(src_dict)))
        logger.info('["sql"] dictionary: {} types'.format(len(sql_dict)))

        return cls(args, src_dict, sql_dict)

 

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(os.pathsep)
        print(paths)
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]
        

        # infer langcode
        src = 'input'
        sql = 'out'

        self.datasets[split] = load_seq_sql_dataset(
            data_path, split, src, self.src_dict, sql, self.sql_dict,  
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            truncate_source=self.args.truncate_source,
            prepend_bos=self.args.add_bos_token
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def build_model(self, args):
        '''
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator(Namespace(**gen_args))
        '''
        return super().build_model(args)

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.sql_dict

    @property
    def col_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.col_dict
