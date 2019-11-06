import tensorflow as tf
import numpy as np
from peach.tf_nn.mtsa import mask_v2
from peach.tf_nn.model_build import ModelStructure
from peach.utils.hparams import HParams
from peach.bert.modeling import BertConfig, BertModel, get_shape_list, get_assignment_map_from_checkpoint
from abc import ABCMeta, abstractmethod
import json
import copy
import six
import os
import logging


class ModelBertAbstractCls(ModelStructure, metaclass=ABCMeta):
    @staticmethod
    def get_default_model_parameters():
        return HParams(
            # todo: add bert params here

            # vocab_size=999,
            # hidden_size=768,
            # num_hidden_layers=12,
            # num_attention_heads=12,
            # intermediate_size=3072,
            # hidden_act="gelu",
            # hidden_dropout_prob=0.1,
            # attention_probs_dropout_prob=0.1,
            # max_position_embeddings=512,
            # type_vocab_size=16,
            # initializer_range=0.03,

            warmup_proportion=0.1,
            learning_rate=5e-5,
        )

    @staticmethod
    def get_identity_param_list():
        return []

    def __init__(self, cfg, is_paired_data, scope, **kwargs):
        self.cfg = cfg
        self.is_paired_data = is_paired_data
        self.scope = scope

        if os.path.isfile(cfg['bert_config_file']):
            self.bert_cfg = BertConfig.from_json_file(cfg['bert_config_file'])
        else:
            raise AttributeError

        # use parsed params
        for key in self.bert_cfg.__dict__.keys():
            if key in cfg:
                setattr(self.bert_cfg, key, cfg[key])

        # use input kwgs
        for key in self.bert_cfg.__dict__.keys():
            if key in kwargs and kwargs[key] is not None:
                setattr(self.bert_cfg, key, kwargs[key])

        # placeholder
        self.is_training = tf.placeholder(tf.bool, [], 'is_training')
        self.input_ids = tf.placeholder(tf.int32, [None, None], 'input_ids')  # bs,sl
        self.input_type_ids = tf.placeholder(tf.int32, [None, None], 'input_type_ids')  # bs, sl

        # extension
        self.bs, self.sl = get_shape_list(self.input_ids)
        self.input_mask = tf.cast(self.input_ids, tf.bool)  # bs,sl
        self.input_len = tf.reduce_sum(tf.cast(self.input_mask, tf.int32), axis=-1)  # bs

        # other placeholder
        self.s1_ids = tf.placeholder(tf.int32, [None, None], 's1_ids')  # bs,sl1
        self.s2_ids = tf.placeholder(tf.int32, [None, None], 's2_ids')  # bs,sl2

        self.sc_ids = tf.placeholder(tf.int32, [None, None], 'sc_ids')  # bs,slc
        self.sc_type_ids = tf.placeholder(tf.int32, [None, None], 'sc_type_ids')  # bs,slc

        # other extension
        self.sl1, self.sl2 = get_shape_list(self.s1_ids)[-1], get_shape_list(self.s2_ids)[-1]
        self.s1_mask = tf.cast(self.s1_ids, tf.bool)
        self.s2_mask = tf.cast(self.s2_ids, tf.bool)
        self.s1_len = tf.reduce_sum(tf.cast(self.s1_mask, tf.int32), axis=-1)  # bs
        self.s2_len = tf.reduce_sum(tf.cast(self.s2_mask, tf.int32), axis=-1)  # bs

        self.slc = get_shape_list(self.sc_ids)[-1]
        self.sc_mask = tf.cast(self.sc_ids, tf.bool)
        self.sc_len = tf.reduce_sum(tf.cast(self.sc_mask, tf.int32), axis=-1)  # bs

        # build bert base model
        with tf.variable_scope('bert_container'):
            self.bert_scp = tf.get_variable_scope().name

            self.input_bert = BertModel(
                self.bert_cfg,
                is_training=self.is_training,
                input_ids=self.input_ids,
                input_mask=tf.cast(self.input_mask, tf.int32),
                token_type_ids=self.input_type_ids,
                use_one_hot_embeddings=False,
                scope="bert",
            )
            tf.get_variable_scope().reuse_variables()
            self.s1_bert = BertModel(
                self.bert_cfg,
                is_training=self.is_training,
                input_ids=self.s1_ids,
                input_mask=tf.cast(self.s1_mask, tf.int32),
                token_type_ids=tf.zeros_like(self.s1_ids),
                use_one_hot_embeddings=False,
                scope="bert",
            )
            self.s2_bert = BertModel(
                self.bert_cfg,
                is_training=self.is_training,
                input_ids=self.s2_ids,
                input_mask=tf.cast(self.s2_mask, tf.int32),
                token_type_ids=tf.ones_like(self.s2_ids),
                use_one_hot_embeddings=False,
                scope="bert",
            )
            self.sc_bert = BertModel(
                self.bert_cfg,
                is_training=self.is_training,
                input_ids=self.sc_ids,
                input_mask=tf.cast(self.sc_mask, tf.int32),
                token_type_ids=self.sc_type_ids,
                use_one_hot_embeddings=False,
                scope="bert",
            )
            self.sc_sl1, self.sc_s1_mask, self.sc_s1_features, self.sc_sl2, self.sc_s2_mask, self.sc_s2_features, \
                = self._split_concatenated_bert_seq(
                    self.sc_bert.get_sequence_output(), self.sc_mask, self.sc_type_ids
                )

    @staticmethod
    def _split_concatenated_bert_seq(input_features, input_mask, token_type_ids):
        with tf.name_scope("split_concatenated_bert_seq"):
            bs, sl, _ = get_shape_list(input_features, expected_rank=3)
            input_mask_int = tf.cast(input_mask, tf.int32)

            # 1. seq: 1
            # 1.1 need a mask
            s1_mask_int = (1 - token_type_ids) * input_mask_int  # bs,sl
            s1_lens = tf.reduce_sum(s1_mask_int, -1)
            sl1 = tf.reduce_max(s1_lens)  # [bs,sl] -> [bs] -> []  # run
            s1_mask = tf.cast(s1_mask_int[:, :sl1], tf.bool)  # bs,sl1
            s1_features = mask_v2(input_features[:, :sl1, :], s1_mask, high_dim=True)  # bs,sl2,hn

            # 2. seq: 2
            s2_lens = tf.reduce_sum(token_type_ids, -1)  # bs
            sl2 = tf.reduce_max(s2_lens)  # []

            # # mask
            sl2_coord = tf.tile(tf.expand_dims(tf.range(sl2, dtype=tf.int32), axis=0), [bs, 1])  # bs,sl2
            s2_mask = tf.less(sl2_coord, tf.tile(tf.expand_dims(s2_lens, axis=1), [1, sl2]))  # bs, sl2

            # # coordination generation
            bs_coord = tf.tile(tf.expand_dims(tf.range(bs, dtype=tf.int32), axis=1), [1, sl2])  # bs,sl2
            seq_coord = sl2_coord + tf.expand_dims(s1_lens, axis=1)  # bs,sl2
            data_indices = tf.stack([bs_coord, seq_coord], axis=-1)  # bs,sl2,2
            data_indices = mask_v2(data_indices, s2_mask, high_dim=True)
            s2_features = tf.gather_nd(input_features, data_indices)  # bs,sl2,hn
            s2_features = mask_v2(s2_features, s2_mask, high_dim=True)  # bs,sl2,hn

            return sl1, s1_mask, s1_features, sl2, s2_mask, s2_features

    def get_feed_dict(self, example_batch, is_train_bool):
        raise NotImplementedError
        # feed_dict = {}
        # # max len in this batch
        # sl = 0
        # for example in example_batch:
        #     sl = max(sl, example.sl)
        #
        # # single input
        # input_ids_list = []
        # input_type_ids_list = []
        # for example in example_batch:
        #     input_ids_snp = np.zeros([sl], dtype=self.cfg['intX'])
        #     input_type_ids_snp = np.zeros([sl], dtype=self.cfg['intX'])
        #     for idx_t, (input_id, segment_id) in enumerate(zip(example.input_ids, example.input_type_ids)):
        #         input_ids_snp[idx_t] = input_id
        #         input_type_ids_snp[idx_t] = segment_id
        #     input_ids_list.append(input_ids_snp)
        #     input_type_ids_list.append(input_type_ids_snp)
        # input_ids_np = np.stack(input_ids_list, 0)
        # input_type_ids_np = np.stack(input_type_ids_list, 0)
        # feed_dict[self.input_ids] = input_ids_np
        # feed_dict[self.input_type_ids] = input_type_ids_np
        #
        # if self.is_paired_data:
        #     # seperate input
        #     sl1, sl2 = 0, 0
        #     for example in example_batch:
        #         sl1 = max(sl1, example.sl1)
        #         sl2 = max(sl2, example.sl2)
        #
        #     s1_ids_list = []
        #     s2_ids_list = []
        #     for example in example_batch:
        #         s1_ids_snp = np.zeros([sl1], dtype=self.cfg['intX'])
        #         for idx_t, s1_id in enumerate(example.s1_ids):
        #             s1_ids_snp[idx_t] = s1_id
        #         s1_ids_list.append(s1_ids_snp)
        #
        #         s2_ids_snp = np.zeros([sl2], dtype=self.cfg['intX'])
        #         for idx_t, s2_id in enumerate(example.s2_ids):
        #             s2_ids_snp[idx_t] = s2_id
        #         s2_ids_list.append(s2_ids_snp)
        #     s1_ids_np = np.stack(s1_ids_list, 0)
        #     s2_ids_np = np.stack(s2_ids_list, 0)
        #     feed_dict[self.s1_ids] = s1_ids_np
        #     feed_dict[self.s2_ids] = s2_ids_np
        #
        #     # combined input
        #     slc = 0
        #     for example in example_batch:
        #         slc = max(slc, example.slc)
        #     sc_ids_list = []
        #     sc_type_ids_list = []
        #     for example in example_batch:
        #         sc_ids_snp = np.zeros([slc], dtype=self.cfg['intX'])
        #         sc_type_ids_snp = np.zeros([slc], dtype=self.cfg['intX'])
        #         for idx_t, (sc_id, segment_id) in enumerate(zip(example.sc_ids, example.sc_type_ids)):
        #             sc_ids_snp[idx_t] = sc_id
        #             sc_type_ids_snp[idx_t] = segment_id
        #         sc_ids_list.append(sc_ids_snp)
        #         sc_type_ids_list.append(sc_type_ids_snp)
        #     sc_ids_np = np.stack(sc_ids_list, 0)
        #     sc_type_ids_np = np.stack(sc_type_ids_list, 0)
        #     feed_dict[self.sc_ids] = sc_ids_np
        #     feed_dict[self.sc_type_ids] = sc_type_ids_np
        #
        # # label
        # if hasattr(example_batch[0], 'label_id'):
        #     gold_label_b = []
        #     for example in example_batch:
        #         gold_label_b.append(example.label_id)
        #     gold_label_b = np.stack(gold_label_b).astype(self.cfg['intX'])
        #     feed_dict[self.gold_label] = gold_label_b
        #
        # # is train
        # feed_dict[self.is_training] = is_train_bool
        #
        # return feed_dict

    def load_pretrained_model_init(self, num_layers=None):
        if num_layers is not None and num_layers == -2:
            logging.info("The num_layers is set to \"-2\", so the BERT loading is disabled!!!")
            return

        if 'init_checkpoint' in self.cfg and tf.train.checkpoint_exists(self.cfg['init_checkpoint']):
            logging.info("Loading bert from {} with num of layers {}".format(self.cfg['init_checkpoint'], num_layers))
            vars = tf.trainable_variables(self.scope)
            assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(
                vars, self.cfg['init_checkpoint'],self.bert_scp)
            # modify the assignment_map for the compatibility with this class
            # final_assignment_map = {}
            # for key in assignment_map.keys():
            #
            #     if isinstance(num_layers, int) and num_layers >= 0:
            #         if num_layers == 0:
            #             if not key.startswith("bert/embeddings"):
            #                 continue
            #         else:
            #             raise NotImplementedError
            #
            #     final_assignment_map[key] = self.bert_scp + '/' + assignment_map[key]
            if num_layers is not None and num_layers >= 0:
                new_assignment_map = {}
                for key in assignment_map.keys():
                    if num_layers == 0:
                        if key.startswith("bert/embeddings/word_embeddings"):
                            new_assignment_map[key] = assignment_map[key]
                    elif num_layers == 1:
                        if key.startswith("bert/embeddings"):
                            new_assignment_map[key] = assignment_map[key]
                assignment_map = new_assignment_map

            for key in assignment_map.keys():
                assignment_map[key] = self.bert_scp + '/' + assignment_map[key]

            tf.train.init_from_checkpoint(self.cfg['init_checkpoint'], assignment_map)
        else:
            logging.warning("no pretrained model found")

    def log_num_params(self):
        # log the number of parameters
        tvars = tf.trainable_variables(self.scope)
        all_params_num = 0
        for elem in tvars:
            params_num = 1
            for l in elem.get_shape().as_list(): params_num *= l
            all_params_num += params_num
        logging.info('Trainable Parameters Number: %d' % all_params_num)

# Commented codes from bert repo
# log_probs = tf.nn.log_softmax(self.logits, axis=-1)
        # one_hot_labels = tf.one_hot(self.gold_label, depth=self.num_labels, dtype=tf.float32)
        # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        # loss = tf.reduce_mean(per_example_loss)
        # return per_example_loss, loss


        # output_layer = self.input_bert.get_pooled_output()
        # hidden_size = output_layer.shape[-1].value
        # output_weights = tf.get_variable(
        #     "output_weights", [self.num_labels, hidden_size],
        #     initializer=tf.truncated_normal_initializer(stddev=0.02))
        # output_bias = tf.get_variable(
        #     "output_bias", [self.num_labels], initializer=tf.zeros_initializer())
        # output_layer = dropout(output_layer, 0.1, self.is_training)
        # logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        # logits = tf.nn.bias_add(logits, output_bias)




