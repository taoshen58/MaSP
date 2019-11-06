import tensorflow as tf
import numpy as np
from peach.bert.model_bert_abs import ModelBertAbstractCls
from peach.utils.hparams import HParams, merge_hparams
from peach.tf_nn.nn import pooling_with_mask, bn_dense_layer_v2, smoothed_softmax_cross_entropy_with_logits
from peach.bert import optimization
from peach.bert.tokenization import convert_tokens_to_ids
from peach.tf_nn.general import mask_v3, get_shape_list, get_last_state, VERY_NEGATIVE_NUMBER
from peach.tf_nn.data.srl import get_word_level_split, mask_matrix_to_coordinate, top_k_to_coordinate
from peach.tf_nn.data.srl import transform_pos_ids_to_wordpiece_idx, generate_mask_based_on_lens
from peach.bert.utils import SPECIAL_TOKENS, SPECIAL_TOKEN_MAPPING
from nn_utils.nn import s2t_self_attn, transformer_seq_decoder
from nn_utils.general import get_key_indices, get_slice
from e2e.e2e_dataset import BaseProcessor
from peach.bert.modeling import BertConfig, BertModel
from e2e.models.bert.logits_build import logits_for_sketch_index, logits_for_sketch_prediction
import collections
import logging

PAD_TOKEN = SPECIAL_TOKENS["PAD"]
UNK_TOKEN = SPECIAL_TOKENS["UNK"]
EMPTY_TOKEN = SPECIAL_TOKENS["EMPTY"]
SOS_TOKEN = SPECIAL_TOKENS["SOS"]
EOS_TOKEN = SPECIAL_TOKENS["EOS"]


class ModelBertTemplate(ModelBertAbstractCls):
    LOSS_GAIN_DICT = {
        "Simple Question (Direct)": 1.,
        "Simple Question (Ellipsis)": 2.8,
        "Quantitative Reasoning (All)": 3.5,
        "Quantitative Reasoning (Count) (All)": 2.,
        "Logical Reasoning (All)": 2.5,
        "Simple Question (Coreferenced)": 1.5,
        "Verification (Boolean) (All)": 2.3,
        "Comparative Reasoning (Count) (All)": 4,
        "Comparative Reasoning (All)": 4,
        "Clarification": 1.,
    }

    @staticmethod
    def get_default_model_parameters():
        return merge_hparams(
            ModelBertAbstractCls.get_default_model_parameters(),
            HParams(
                clf_act_name='gelu',
                clf_dropout=0.1,
                clf_attn_dropout=0.1,
                clf_res_dropout=0.1,
                # clf_head_num=12,
                pos_gain=8.0,
                pos_threshold=0.5,
                hn=768,
                # label_smoothing=0.1,
                decoder_layer=3,
                clf_head_num=12,
                use_qt_loss_gain=False,
                seq_label_loss_weight=1.,
                seq2seq_loss_weight=1.,

                level_for_ner=4,
                level_for_predicate=3,
                level_for_type=5,
                level_for_dec=-1,
                hidden_size_input=-1,
                num_attention_heads_input=-1,
                intermediate_size_input=-1,
                hidden_dropout_prob_input=-1.,
                attention_probs_dropout_prob_input=-1.,
            )
        )

    @staticmethod
    def get_identity_param_list():
        return ModelBertAbstractCls.get_identity_param_list() + []

    def __init__(self, cfg, tokenizer, data_type, labels_dict, max_sequence_len, num_training_steps, scope):
        if "level_for_dec" in cfg and cfg['level_for_dec'] >= 0:
            num_hidden_layers = cfg['level_for_dec'] + 1
        else:
            num_hidden_layers = None

        if "hidden_size_input" in cfg and cfg['hidden_size_input'] > 0:
            hidden_size = cfg['hidden_size_input']
        else:
            hidden_size = None

        if "num_attention_heads_input" in cfg and cfg['num_attention_heads_input'] > 0:
            num_attention_heads = cfg['num_attention_heads_input']
        else:
            num_attention_heads = None

        if "intermediate_size_input" in cfg and cfg['intermediate_size_input'] > 0:
            intermediate_size = cfg['intermediate_size_input']
        else:
            intermediate_size = None

        if "hidden_dropout_prob_input" in cfg and cfg['hidden_dropout_prob_input'] > 0:
            hidden_dropout_prob = cfg['hidden_dropout_prob_input']
        else:
            hidden_dropout_prob = None

        if "attention_probs_dropout_prob_input" in cfg and cfg['attention_probs_dropout_prob_input'] > 0:
            attention_probs_dropout_prob = cfg['attention_probs_dropout_prob_input']
        else:
            attention_probs_dropout_prob = None

        super(ModelBertTemplate, self).__init__(
            cfg, is_paired_data=False, scope=scope, num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size, num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob, attention_probs_dropout_prob=attention_probs_dropout_prob,
        )

        self.data_type = data_type
        self.labels_dict = labels_dict
        self.max_sequence_len = max_sequence_len
        self.num_training_steps = num_training_steps
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer

        self.input_pos_ids = tf.placeholder(tf.int32, [None, None])
        self.loss_gain_wrt_qt = tf.placeholder(tf.float32, [None])
        # ==== an introduction to lengths =====
        # [prev_q] [sep] [prev_a] [sep1] [cur_q] [cls]
        # sl: seq len, wordpiece-level, ([prev_q] [sep] [prev_a] [sep1] [cur_q] [cls])
        # sll: seq label len, token-level, ([prev_q] [sep] [prev_a] [sep1] [cur_q])
        # asl: all seq len, token_level, ([prev_q] [sep] [prev_a] [sep1] [cur_q] [cls])
        # # others,
        # pl, piece len, the max len of word pieces belonging to a word

        # ====== labels =====
        # 1. EO
        self.num_EO_labels = len(labels_dict["EOs"]["labels"])
        self.num_type_labels = len(labels_dict["types"]["labels"])
        self.EO_label = tf.placeholder(tf.int32, [None, None])  # [bs, sll] with [0,nel)
        self.entity_type_label = tf.placeholder(tf.int32, [None, None])  # [bs, sll] with (1,ntl)

        # 2. Sketches: include sketch itself and leaves labels: entity, predicate, type and num
        self.sos_id = labels_dict["sketch"]["labels"].index(SOS_TOKEN)
        self.eos_id = labels_dict["sketch"]["labels"].index(EOS_TOKEN)
        self.num_predicate_labels = len(labels_dict["predicates"]["labels"])
        self.num_sketch_labels = len(labels_dict["sketch"]["labels"])
        self.sketch_label = tf.placeholder(tf.int32, [None, None])  # bs,dsl+1
        self.sketch_output_ids = self.sketch_label[:, 1:]  # bs,dsl
        self.sketch_mask = tf.cast(self.sketch_output_ids, tf.bool)  # bs,dsl
        self.sketch_input_ids = self.sketch_label[:, :-1] * tf.cast(self.sketch_mask, tf.int32)  # bs,dsl
        self.sketch_entity = tf.placeholder(tf.int32, [None, None])  # bs,dsl
        self.sketch_predicate = tf.placeholder(tf.int32, [None, None])  # bs,dsl
        self.sketch_type = tf.placeholder(tf.int32, [None, None])  # bs,dsl
        self.sketch_num = tf.placeholder(tf.int32, [None, None])  # bs,dsl
        # # 2.1 masks
        self.sketch_entity_mask = tf.not_equal(self.sketch_entity, -1)
        self.sketch_predicate_mask = tf.not_equal(self.sketch_predicate, 0)
        self.sketch_type_mask = tf.not_equal(self.sketch_type, 0)
        self.sketch_num_mask = tf.not_equal(self.sketch_num, -1)

        # lens
        self.asl = tf.reduce_max(self.input_pos_ids) + 1  # all sequence length (token-level)
        self.sll = get_shape_list(self.EO_label)[-1]  # sequence labeling length (token-level)
        self.wordpiece_idx = transform_pos_ids_to_wordpiece_idx(  # bs,asl
            self.input_pos_ids, self.input_mask, self.asl)
        self.pl = tf.reduce_max(self.wordpiece_idx) + 1

        # masks
        self.seq_label_mask = tf.cast(self.EO_label, bool)  # bs,sll
        self.wordpiece_mask = get_word_level_split(  # bs,asl,pl
            self.input_mask, self.input_pos_ids, self.wordpiece_idx, self.input_mask, self.asl, self.pl
        )

        # special token indices
        self.unk_id, self.cls_id, self.sep_id, self.empty_id, self.sep1_id, self.pad_id = convert_tokens_to_ids(
            self.vocab,
            [
                SPECIAL_TOKENS["UNK"], SPECIAL_TOKENS["CLS"], SPECIAL_TOKENS["SEP"],
                SPECIAL_TOKENS["EMPTY"], SPECIAL_TOKENS["SEP1"], SPECIAL_TOKENS["PAD"]])

        # for the decoding
        self.dec_input_emb_mat = tf.get_variable(
            "dec_input_emb_mat", [self.num_sketch_labels, self.cfg["hn"]],
            initializer=tf.truncated_normal_initializer(0, 0.05)
        )

        # for the key indices
        first_ids = get_word_level_split(  # bs,sl -> bs,asl,pl -> bs,asl
            self.input_ids, self.input_pos_ids, self.wordpiece_idx, self.input_mask, self.asl, self.pl
        )[..., 0]  # get the 1st id in each wordpieces
        self.sep_indices = tf.stack(
            get_key_indices(first_ids, [self.sep_id, self.sep1_id, self.cls_id]), axis=-1)

        self.decoder_dict = {
            # placeholders: don't forget the
            "encoder_states_placeholder": tf.placeholder(tf.float32, [None, None, cfg["hn"]]),  # bs,sl,hn
            "encoder_output_for_predicate_placeholder": tf.placeholder(tf.float32, [None, cfg["hn"]]),
            "encoder_output_for_type_placeholder": tf.placeholder(tf.float32, [None, cfg["hn"]]),
            "encoder_ids_placeholder": tf.placeholder(tf.float32, [None, None]),  # bs,sl
            "decoder_history_placeholder": tf.placeholder(tf.float32, [None, cfg["decoder_layer"], None, cfg["hn"]]),
            # bs,t,hn
            "decoder_ids_placeholder": tf.placeholder(tf.int32, [None, 1]),
            "is_training_placeholder": self.is_training,
            # intermediate tensor
            "encoder_states_run": None,
            "encoder_output_for_predicate_run": None,
            "encoder_output_for_type_run": None,
            "decoder_history_run": None,
            "logits_seq2seq_run": None,
            "logits_sketch_entity_run": None,
            "logits_sketch_predicate_run": None,
            "logits_sketch_type_run": None,
            "logits_sketch_num_run": None,
        }
        self.decoder_dict["encoder_mask"] = tf.cast(self.decoder_dict["encoder_ids_placeholder"], tf.bool)

        self.logits_dict = None
        self.loss_dict = None
        self.prediction_dict = None

        self.loss = None
        self.train_op = None

        self.run_dict = None

        self._setup_training()

    def step(self, sess, example_batch):
        feed_dict = self.get_feed_dict(example_batch, True)
        run_res = sess.run(self.run_dict, feed_dict=feed_dict)
        run_res.pop("train_op")
        return run_res

    def get_init_decoder_history_np(self, bs_init):
        history_np = np.zeros([bs_init, self.cfg["decoder_layer"], 0, self.cfg["hn"]], "float32")
        return history_np

    def get_init_decoder_ids_np(self, bs_init):
        decoder_ids_np = np.ones([bs_init, 1], "int32") * self.sos_id
        return decoder_ids_np

    def _setup_training(self):
        self.logits_dict = self._build_network()
        self.loss, self.loss_dict = self._build_loss()
        self.prediction_dict = self._build_prediction()

        self.log_num_params()

        # to build train op
        self.train_op = optimization.create_optimizer(
            self.loss,
            self.cfg['learning_rate'],
            self.num_training_steps,
            int(self.num_training_steps * self.cfg['warmup_proportion']),
            use_tpu=False
        )

        self.run_dict = {
            "loss": self.loss,
            "loss_seq2seq": self.loss_dict["seq2seq"],
            "loss_seq_label": self.loss_dict["seq_label"],
            "train_op": self.train_op,
        }

        # for decoder beam search
        # # 1. for distribution
        seq2seq_dist_wo_pad = tf.nn.softmax(self.decoder_dict["logits_seq2seq_run"])  # bs,1,nl-1
        self.decoder_dict["distribution_seq2seq_run"] = tf.concat(  # bs,1,nl
            [
                tf.zeros(get_shape_list(seq2seq_dist_wo_pad)[:2] + [1]),  # bs,1,1
                seq2seq_dist_wo_pad,
            ], -1)
        self.decoder_dict["distribution_sketch_entity_run"] = tf.nn.softmax(
            self.decoder_dict["logits_sketch_entity_run"])
        self.decoder_dict["distribution_sketch_predicate_run"] = tf.nn.softmax(
            self.decoder_dict["logits_sketch_predicate_run"])
        self.decoder_dict["distribution_sketch_type_run"] = tf.nn.softmax(
            self.decoder_dict["logits_sketch_type_run"])
        self.decoder_dict["distribution_sketch_num_run"] = tf.nn.softmax(
            self.decoder_dict["logits_sketch_num_run"])

    def _build_network(self):
        level_for_ner = self.cfg["level_for_ner"] if "level_for_ner" in self.cfg else 4
        level_for_predicate = self.cfg["level_for_predicate"] if "level_for_predicate" in self.cfg else 3
        level_for_type = self.cfg["level_for_type"] if "level_for_type" in self.cfg else 5
        level_for_dec = self.cfg["level_for_dec"] if "level_for_dec" in self.cfg else -1
        logging.info("level_for_ner is {}, level_for_predicate is {}, level_for_type is {}, level_for_dec is {}".format(
            level_for_ner, level_for_predicate, level_for_type, level_for_dec))

        all_encoder_states = self.input_bert.get_all_encoder_layers()
        encoder_states_for_seq_label = all_encoder_states[level_for_ner]
        encoder_states_for_decoder = all_encoder_states[level_for_dec]
        encoder_output_for_predicate = get_last_state(all_encoder_states[level_for_predicate], self.input_mask)
        encoder_output_for_type = get_last_state(all_encoder_states[level_for_type], self.input_mask)
        # EO+ent_type logits
        logits_seq_label = self._build_network_seq_label_logits(encoder_states_for_seq_label)  # bs,sll,xxx

        with tf.variable_scope("decoder_container"):
            train_out_states, train_logits, _ = transformer_seq_decoder(  #
                self.dec_input_emb_mat, self.sketch_input_ids, encoder_states_for_decoder,
                self.sketch_mask, self.input_mask, self.num_sketch_labels - 1, self.cfg["decoder_layer"], None,
                self.cfg["hn"], self.cfg["clf_head_num"], self.cfg["clf_act_name"], 0., self.is_training,
                1 - self.cfg["clf_dropout"], 1 - self.cfg["clf_attn_dropout"],
                1 - self.cfg["clf_res_dropout"], scope="transformer_seq_decoder"
            )
            logits_sketch_entity, logits_sketch_predicate, logits_sketch_type, logits_sketch_num = \
                self._build_network_all_sketch_logits(
                    train_out_states, encoder_states_for_decoder, self.input_mask,
                    encoder_output_for_predicate, encoder_output_for_type, use_mask=True
            )
            tf.get_variable_scope().reuse_variables()
            infer_out_states, infer_logits, new_decoder_history_inputs = transformer_seq_decoder(  # [bs,1]
                self.dec_input_emb_mat, self.decoder_dict["decoder_ids_placeholder"],
                self.decoder_dict["encoder_states_placeholder"],
                tf.cast(self.decoder_dict["decoder_ids_placeholder"], tf.bool),
                self.decoder_dict["encoder_mask"],
                self.num_sketch_labels - 1, self.cfg["decoder_layer"],
                self.decoder_dict["decoder_history_placeholder"],
                self.cfg["hn"], self.cfg["clf_head_num"], self.cfg["clf_act_name"], 0., self.is_training,
                1 - self.cfg["clf_dropout"], 1 - self.cfg["clf_attn_dropout"], 1 - self.cfg["clf_res_dropout"],
                scope="transformer_seq_decoder"
            )
            self.decoder_dict["logits_sketch_entity_run"], self.decoder_dict["logits_sketch_predicate_run"], \
                    self.decoder_dict["logits_sketch_type_run"], self.decoder_dict["logits_sketch_num_run"] = \
                self._build_network_all_sketch_logits(
                    infer_out_states, self.decoder_dict["encoder_states_placeholder"],
                    self.decoder_dict["encoder_mask"],
                    self.decoder_dict["encoder_output_for_predicate_placeholder"],
                    self.decoder_dict["encoder_output_for_type_placeholder"]
            )
            self.decoder_dict["encoder_states_run"] = encoder_states_for_decoder
            self.decoder_dict["encoder_output_for_predicate_run"] = encoder_output_for_predicate
            self.decoder_dict["encoder_output_for_type_run"] = encoder_output_for_type
            self.decoder_dict["decoder_history_run"] = new_decoder_history_inputs
            self.decoder_dict["logits_seq2seq_run"] = infer_logits

            return {
                "seq_label": logits_seq_label,
                "seq2seq": train_logits,
                "sketch_entity": logits_sketch_entity,
                "sketch_predicate": logits_sketch_predicate,
                "sketch_type": logits_sketch_type,
                "sketch_num": logits_sketch_num,
            }

    def _build_network_all_sketch_logits(
            self, decoder_states, encoder_states_for_decoder,
            encoder_mask, cls_state_predicate, cls_state_type,
            use_mask=False
    ):
        bs = get_shape_list(decoder_states)[0]
        if use_mask:
            entity_mask = tf.not_equal(self.sketch_entity, -1)
            predicate_mask = tf.not_equal(self.sketch_predicate, 0)
            type_mask = tf.not_equal(self.sketch_type, 0)
            num_mask = tf.not_equal(self.sketch_num, -1)
        else:
            entity_mask = None
            predicate_mask = None
            type_mask = None
            num_mask = None
        # bs,sl -----modify the last token to False
        encoder_wo_cls = tf.concat([  # [bs,sl]
            encoder_mask[:, 1:],  # [bs,sl-1]
            tf.cast(tf.zeros([get_shape_list(encoder_mask)[0], 1], tf.int32), tf.bool)  # [bs, 1]
        ], -1)

        logits_sketch_entity_pre = logits_for_sketch_index(  # bs,dsl,esl
            decoder_states, encoder_states_for_decoder, self.cfg["hn"], 0., 1 - self.cfg["clf_dropout"],
            self.is_training, compress_mask=entity_mask, scope="logits_sketch_entity_pre"
        )
        logits_sketch_entity = mask_v3(
            logits_sketch_entity_pre, encoder_wo_cls, multi_head=True, name="logits_sketch_entity")

        logits_sketch_predicate_pre = logits_for_sketch_prediction(
            decoder_states, cls_state_predicate, self.num_predicate_labels - 3,  self.cfg["hn"],
            self.cfg["clf_act_name"], 0., 1 - self.cfg["clf_dropout"], self.is_training,
            compress_mask=predicate_mask,
            scope="logits_sketch_predicate"
        )
        logits_sketch_predicate = tf.concat([
            tf.ones([bs, get_shape_list(logits_sketch_predicate_pre)[1], 3], tf.float32) * VERY_NEGATIVE_NUMBER,
            logits_sketch_predicate_pre,
        ], axis=-1)

        logits_sketch_type_pre = logits_for_sketch_prediction(
            decoder_states, cls_state_type, self.num_type_labels - 3, self.cfg["hn"],
            self.cfg["clf_act_name"], 0., 1 - self.cfg["clf_dropout"], self.is_training,
            compress_mask=type_mask,
            scope="logits_sketch_type"
        )
        logits_sketch_type = tf.concat([
            tf.ones([bs, get_shape_list(logits_sketch_type_pre)[1], 3], tf.float32) * VERY_NEGATIVE_NUMBER,
            logits_sketch_type_pre,
        ], axis=-1)

        logits_sketch_num_pre = logits_for_sketch_index(
            decoder_states, encoder_states_for_decoder, self.cfg["hn"], 0., 1 - self.cfg["clf_dropout"],
            self.is_training,  compress_mask=num_mask, scope="logits_sketch_num_pre"
        )
        logits_sketch_num = mask_v3(
            logits_sketch_num_pre, encoder_wo_cls, multi_head=True, name="logits_sketch_num")

        return logits_sketch_entity, logits_sketch_predicate, logits_sketch_type, logits_sketch_num

    def _build_network_seq_label_logits(self, encoder_states):
        wp_features = get_word_level_split(  # bs,sl,hn -> bs,asl,pl,hn
            encoder_states, self.input_pos_ids, self.wordpiece_idx, self.input_mask, self.asl, self.pl
        )

        all_token_features = s2t_self_attn(  # bs,asl,hn
            wp_features, self.wordpiece_mask, self.cfg['clf_act_name'], 'multi_dim',
            0., 1.-self.cfg['clf_dropout'], self.is_training, 'all_token_features',
        )
        # get seq_label_token_features  asl -> sll (asl-1)
        seq_label_token_features = mask_v3(  # remove the latest feature
            all_token_features[:, :-1], self.seq_label_mask, high_dim=True
        )

        with tf.variable_scope("output"):
            with tf.variable_scope("seq_labeling"):
                seq_label_logits = bn_dense_layer_v2(  # "O"  (NO PAD   for predicate no empty no pad
                    seq_label_token_features, 1 + (self.num_EO_labels-2) * (self.num_type_labels-2),
                    True, 0., "seq_labeling_logits", "linear", False,
                    0., 1. - self.cfg['clf_dropout'], self.is_training
                )
        return seq_label_logits

    def _build_loss(self):
        # for seq label
        joint_label = tf.where(  # 0 for empty or pad
            tf.logical_and(tf.greater_equal(self.EO_label, 2), tf.greater_equal(self.entity_type_label, 2)),
            (self.EO_label - 2) + 4 * (self.entity_type_label - 2) + 1,
            tf.zeros(get_shape_list(self.EO_label), tf.int32)
        )
        joint_label_rsp = tf.reshape(joint_label, [self.bs * self.sll])
        logits_seq_label_rsp = tf.reshape(self.logits_dict["seq_label"],
                                          [-1, get_shape_list(self.logits_dict["seq_label"])[-1]])

        losses_seq_label_rsp = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=joint_label_rsp, logits=logits_seq_label_rsp
        )
        losses_seq_label = tf.reshape(losses_seq_label_rsp, [self.bs, self.sll])
        seq_label_mask_tf = tf.cast(self.seq_label_mask, tf.float32)
        seq_label_weights = tf.where(
            tf.greater(joint_label, 0),
            tf.ones_like(losses_seq_label) * self.cfg["pos_gain"],
            tf.ones_like(losses_seq_label)
        ) * seq_label_mask_tf
        loss_seq_label = \
            tf.reduce_sum(losses_seq_label * seq_label_weights) / tf.reduce_sum(seq_label_weights)

        # for sequence to sequence
        # # 1. sketch loss
        label_seq2seq = tf.where(
            self.sketch_mask,
            self.sketch_output_ids - 1,  # for valid token - 1
            self.sketch_output_ids
        )
        losses_sketch = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_seq2seq,
            logits=self.logits_dict["seq2seq"]
        )

        # # 2. leaves losses
        # # # 2.1 entity
        losses_sketch_entity = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.sketch_entity * tf.cast(self.sketch_entity_mask, tf.int32),
            logits=self.logits_dict["sketch_entity"]
        ) * tf.cast(self.sketch_entity_mask, tf.float32)

        # # # 2.2 predicate
        losses_sketch_predicate = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.sketch_predicate,
            logits=self.logits_dict["sketch_predicate"]
        ) * tf.cast(self.sketch_predicate_mask, tf.float32)
        # # # 2.3 type
        losses_sketch_type = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.sketch_type,
            logits=self.logits_dict["sketch_type"]
        ) * tf.cast(self.sketch_type_mask, tf.float32)
        # # # 2.4 num
        losses_sketch_num = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.sketch_num * tf.cast(self.sketch_num_mask, tf.int32),
            logits=self.logits_dict["sketch_num"]
        ) * tf.cast(self.sketch_num_mask, tf.float32)
        # # 3 combine leaves' losses
        losses_sketch_leaves = \
            (losses_sketch_entity + losses_sketch_predicate + losses_sketch_type + losses_sketch_num) * \
            tf.cast(self.sketch_mask, tf.float32)
        # # 4. combine to the sketch loss
        losses_seq2seq = losses_sketch + losses_sketch_leaves
        # # 5. calc final loss
        sketch_mask_ft = tf.cast(self.sketch_mask, tf.float32)  # bs,sl
        sketch_mask_int = tf.cast(self.sketch_mask, tf.int32)  # bs,sl

        sketch_ex_mask = tf.cast(tf.reduce_sum(sketch_mask_int, -1), tf.bool)  # bs
        sketch_ex_mask_ft = tf.cast(sketch_ex_mask, tf.float32)  # bs

        seq_deno = tf.reduce_sum(sketch_mask_ft, -1)
        seq_deno = tf.where(
            tf.greater(seq_deno, 0.),
            seq_deno,
            tf.ones_like(seq_deno) * 1e-6,
        )
        loss_seq2seq_example = tf.reduce_sum(sketch_mask_ft * losses_seq2seq, -1) / seq_deno  # bs
        loss_seq2seq_example = loss_seq2seq_example * self.loss_gain_wrt_qt

        batch_deno = tf.reduce_sum(sketch_ex_mask_ft * self.loss_gain_wrt_qt)
        batch_deno = tf.where(
            tf.greater(batch_deno, 0.),
            batch_deno,
            tf.ones_like(batch_deno) * 1e-6,
        )
        loss_seq2seq = tf.reduce_sum(sketch_ex_mask_ft * loss_seq2seq_example) / batch_deno

        opt_loss = self.cfg["seq_label_loss_weight"]*loss_seq_label + \
                   self.cfg["seq2seq_loss_weight"] * loss_seq2seq
        return opt_loss, {
            "seq_label": loss_seq_label,
            "seq2seq": loss_seq2seq,
        }

    def _build_prediction(self):
        # # for NER sequence labeling
        predictions_seq_label = tf.cast(tf.argmax(self.logits_dict["seq_label"], axis=-1), tf.int32)
        predictions_ner = tf.where(
            tf.greater_equal(predictions_seq_label, 1),
            tf.mod(predictions_seq_label - 1, 4) + 2,
            tf.ones_like(predictions_seq_label)
        )
        predictions_ner = mask_v3(predictions_ner, self.seq_label_mask)
        # # for predicate
        predictions_entity_type = tf.where(
            tf.greater_equal(predictions_seq_label, 1),
            tf.cast((predictions_seq_label - 1) / 4, tf.int32) + 2,
            tf.ones_like(predictions_seq_label)
        )
        predictions_entity_type = mask_v3(predictions_entity_type, self.seq_label_mask)

        # # for semantic parsing
        predicted_seq2seq = tf.cast(tf.argmax(self.logits_dict["seq2seq"], axis=-1), tf.int32)  # bs,sl
        predicted_seq2seq = tf.where(
            self.sketch_mask,
            predicted_seq2seq + 1,
            tf.zeros_like(predicted_seq2seq)
        )

        predicted_sketch_entity = tf.cast(tf.argmax(self.logits_dict["sketch_entity"], axis=-1), tf.int32)  # bs,sl
        predicted_sketch_entity = tf.where(
            self.sketch_entity_mask,
            predicted_sketch_entity,
            -tf.ones_like(predicted_sketch_entity)
        )

        predicted_sketch_predicate = tf.cast(tf.argmax(self.logits_dict["sketch_predicate"], axis=-1), tf.int32)  # bs,sl
        predicted_sketch_predicate = tf.where(
            self.sketch_predicate_mask,
            predicted_sketch_predicate,
            tf.zeros_like(predicted_sketch_predicate)
        )

        predicted_sketch_type = tf.cast(tf.argmax(self.logits_dict["sketch_type"], axis=-1), tf.int32)  # bs,sl
        predicted_sketch_type = tf.where(
            self.sketch_type_mask,
            predicted_sketch_type,
            tf.zeros_like(predicted_sketch_type)
        )

        predicted_sketch_num = tf.cast(tf.argmax(self.logits_dict["sketch_num"], axis=-1), tf.int32)  # bs,sl
        predicted_sketch_num = tf.where(
            self.sketch_num_mask,
            predicted_sketch_num,
            -tf.ones_like(predicted_sketch_num)
        )

        return {
            "EOs": predictions_ner,
            "entity_types": predictions_entity_type,
            "seq_label_mask": self.seq_label_mask,
            "sketch": predicted_seq2seq,
            "sketch_entity": predicted_sketch_entity,
            "sketch_predicate": predicted_sketch_predicate,
            "sketch_type": predicted_sketch_type,
            "sketch_num": predicted_sketch_num,

            # aux
            "sep_indices": self.sep_indices,
        }

    def get_feed_dict(self, example_batch, is_train_bool, max_seq_len=None):
        max_seq_len = max_seq_len or self.max_sequence_len

        def _get_ids_for_u(_u_ids, _u_pos_ids, _base_pos):
            _u_ids = _u_ids.copy()
            _u_pos_ids = _u_pos_ids.copy()
            for _idx in range(len(_u_pos_ids)):
                _u_pos_ids[_idx] += _base_pos
            if len(_u_ids) == 0 and len(_u_pos_ids) == 0:  # add [EMPTY]
                _u_ids = [self.empty_id]  # convert_tokens_to_ids(self.vocab, [EMPTY_TOKEN])
                _u_pos_ids = [_base_pos]
            assert len(_u_ids) == len(_u_pos_ids)
            _base_pos = _u_pos_ids[-1] + 1
            return _u_ids, _u_pos_ids, _base_pos

        feed_dict = {self.is_training: is_train_bool}
        bs = len(example_batch)

        # add cache in each batch
        for example in example_batch:
            if "cache" not in example:
                example["cache"] = {}

        # for input_ids and input_type_ids
        # # for a sequence [prev_q] [SEP] [prev_a] [SEP1] [cur_q] [CLS]
        # # if len of the digitized_seq is 0, add [EMPTY]
        max_wpl = 0  # max word peace len
        input_ids_list = []
        input_pos_ids_list = []
        input_type_ids_list = []
        u_lens_list = []
        max_sll = 0  # max sequence labeling len
        EO_label_list = []
        entity_type_label_list = []
        # predicate_label_list = []
        # type_label_list = []

        for example in example_batch:
            if "input_ids" in example["cache"]:
                ipt_ids = example["cache"]["input_ids"]
                ipt_pos_ids = example["cache"]["input_pos_ids"]
                ipt_type_ids = example["cache"]["input_type_ids"]
                u_lens = example["cache"]["u_lens"]
            else:
                pos_base = 0
                prev_q_ids, prev_q_pos_ids, pos_base = _get_ids_for_u(
                    example["tokenized_utterances_ids"]["prev_q"],
                    example["tokenized_utterances_pos_ids"]["prev_q"],
                    pos_base
                )
                sep1 = ([self.sep_id], [pos_base])
                pos_base += 1

                prev_a_ids, prev_a_pos_ids, pos_base = _get_ids_for_u(
                    example["tokenized_utterances_ids"]["prev_a"],
                    example["tokenized_utterances_pos_ids"]["prev_a"],
                    pos_base
                )
                sep2 = ([self.sep1_id], [pos_base])
                pos_base += 1

                cur_q_ids, cur_q_pos_ids, pos_base = _get_ids_for_u(
                    example["tokenized_utterances_ids"]["cur_q"],
                    example["tokenized_utterances_pos_ids"]["cur_q"],
                    pos_base
                )
                sep3 = ([self.cls_id], [pos_base])
                pos_base += 1

                all_seq_data = [
                    (prev_q_ids, prev_q_pos_ids),
                    sep1,
                    (prev_a_ids, prev_a_pos_ids),
                    sep2,
                    (cur_q_ids, cur_q_pos_ids),
                    sep3,
                ]
                # add pos_ids mask
                all_pos_ids = []
                for _, _pos_ids in all_seq_data:
                    all_pos_ids.extend(_pos_ids)
                all_pos_ids_mask = pos_ids_mask_gen(all_pos_ids, max_seq_len)
                # # apply mask
                new_all_seq_data = []
                mask_ptr = 0
                for _ids, _pos_ids in all_seq_data:
                    ids_len = len(_ids)
                    ids_mask = all_pos_ids_mask[mask_ptr: mask_ptr+ids_len]
                    assert ids_len == len(_pos_ids)
                    assert ids_len == len(ids_mask)
                    new_ids = [_id for _id, _m in zip(_ids, ids_mask) if _m == 1]
                    new_pos_ids = [_id for _id, _m in zip(_pos_ids, ids_mask) if _m == 1]
                    new_all_seq_data.append((new_ids, new_pos_ids))
                    mask_ptr += ids_len
                all_seq_data = new_all_seq_data

                u_lens = [len(_d[0]) for _d in all_seq_data]
                # filter
                if max_seq_len > 0:
                    while sum(u_lens) > max_seq_len:
                        for _i in range(len(u_lens)):
                            if u_lens[_i] > 1:
                                u_lens[_i] -= 1

                # ids for this batch
                ipt_ids = []
                ipt_pos_ids = []
                ipt_type_ids = []
                for _idx_d, (_d, _len) in enumerate(zip(all_seq_data, u_lens)):
                    ipt_ids.extend(_d[0][:_len])
                    ipt_pos_ids.extend(_d[1][:_len])
                    ipt_type_ids.extend([_idx_d] * _len)
                # re-scale the pos id
                new_ipt_pos_ids = []
                prev_pos_id = None
                new_pos_id = -1
                for pos_id in ipt_pos_ids:
                    if prev_pos_id is None or pos_id != prev_pos_id:
                        new_pos_id += 1
                        prev_pos_id = pos_id
                    new_ipt_pos_ids.append(new_pos_id)
                ipt_pos_ids = new_ipt_pos_ids

                example["cache"]["input_ids"] = ipt_ids
                example["cache"]["input_pos_ids"] = ipt_pos_ids
                example["cache"]["input_type_ids"] = ipt_type_ids
                example["cache"]["u_lens"] = u_lens

            assert len(ipt_ids) == len(ipt_pos_ids) and len(ipt_pos_ids) == len(ipt_type_ids)
            max_wpl = max(max_wpl, len(ipt_ids))
            # append
            input_ids_list.append(ipt_ids)
            input_pos_ids_list.append(ipt_pos_ids)
            input_type_ids_list.append(ipt_type_ids)
            u_lens_list.append(u_lens)

            # ====== labels =====
            # prev_q_len, prev_a_len, cur_q_len = (
            #     len(set(ipt_pos_ids[:u_lens[0]])),
            #     len(set(ipt_pos_ids[sum(u_lens[:2]):sum(u_lens[:3])])),
            #     len(set(ipt_pos_ids[sum(u_lens[:4]):sum(u_lens[:5])]))
            # )  # u_lens[2], u_lens[4]
            prev_q_len = ipt_pos_ids[u_lens[0]-1] + 1
            prev_a_len = ipt_pos_ids[sum(u_lens[:3])-1] - ipt_pos_ids[sum(u_lens[:2])] + 1
            cur_q_len = ipt_pos_ids[sum(u_lens[:5])-1] - ipt_pos_ids[sum(u_lens[:4])] + 1

            # # EO labels
            prev_q_EOs = [self.empty_id] if len(example["EOs_ids"]["prev_q"]) == 0 else example["EOs_ids"]["prev_q"]
            prev_a_EOs = [self.empty_id] if len(example["EOs_ids"]["prev_a"]) == 0 else example["EOs_ids"]["prev_a"]
            cur_q_EOs = [self.empty_id] if len(example["EOs_ids"]["cur_q"]) == 0 else example["EOs_ids"]["cur_q"]

            assert prev_q_len <= len(prev_q_EOs), "{}_{}".format(prev_q_len, prev_q_EOs)
            assert prev_a_len <= len(prev_a_EOs), "{}_{}".format(prev_a_len, prev_a_EOs)
            assert cur_q_len <= len(cur_q_EOs), "{}_{}".format(cur_q_len, cur_q_EOs)

            EO_label = prev_q_EOs[:prev_q_len] + [self.empty_id] + prev_a_EOs[:prev_a_len] + \
                       [self.empty_id] + cur_q_EOs[:cur_q_len]
            # # Entity Types
            prev_q_entity_types = [self.empty_id] if len(example["entity_types_ids"]["prev_q"]) == 0 else \
                example["entity_types_ids"]["prev_q"]
            prev_a_entity_types = [self.empty_id] if len(example["entity_types_ids"]["prev_a"]) == 0 else \
                example["entity_types_ids"]["prev_a"]
            cur_q_entity_types = [self.empty_id] if len(example["entity_types_ids"]["cur_q"]) == 0 else \
                example["entity_types_ids"]["cur_q"]

            entity_type_label = prev_q_entity_types[:prev_q_len] + [self.empty_id] + prev_a_entity_types[:prev_a_len] \
                                + [self.empty_id] + cur_q_entity_types[:cur_q_len]
            # # # predicates
            # predicate_label = example["predicates_ids"]["cur_q"]
            # # # types
            # type_label = example["types_ids"]["cur_q"]

            # sanity
            assert len(EO_label) == len(entity_type_label)
            assert len(EO_label) == ipt_pos_ids[-1] - ipt_pos_ids[0] + 1 - 1
            max_sll = max(max_sll, len(EO_label))

            EO_label_list.append(EO_label)
            entity_type_label_list.append(entity_type_label)
            # predicate_label_list.append(predicate_label)
            # type_label_list.append(type_label)

        # ====== create ======
        input_ids_np = np.zeros([bs, max_wpl, ], dtype=self.cfg["intX"])
        input_pos_ids_np = np.zeros([bs, max_wpl, ], dtype=self.cfg["intX"])
        input_type_ids_np = np.zeros([bs, max_wpl, ], dtype=self.cfg["intX"])
        EO_label_np = np.zeros([bs, max_sll, ], dtype=self.cfg["intX"])
        entity_type_label_np = np.zeros([bs, max_sll, ], dtype=self.cfg["intX"])
        # predicate_label_np = np.zeros([bs, self.num_predicate_labels, ], dtype=self.cfg["intX"])
        # type_label_np = np.zeros([bs, self.num_type_labels, ], dtype=self.cfg["intX"])

        for idx_e, example in enumerate(example_batch):
            for idx_wp, (_id, _pos_id, _type_id) in enumerate(
                    zip(input_ids_list[idx_e], input_pos_ids_list[idx_e], input_type_ids_list[idx_e])):
                input_ids_np[idx_e, idx_wp] = _id
                input_pos_ids_np[idx_e, idx_wp] = _pos_id
                input_type_ids_np[idx_e, idx_wp] = _type_id
            for idx_sl, (_eo_id, _et_id) in enumerate(zip(EO_label_list[idx_e], entity_type_label_list[idx_e])):
                EO_label_np[idx_e, idx_sl] = _eo_id
                entity_type_label_np[idx_e, idx_sl] = _et_id

            # for _predicate_idx in predicate_label_list[idx_e]:
            #     predicate_label_np[idx_e, _predicate_idx] = 1
            # for _type_idx in type_label_list[idx_e]:
            #     type_label_np[idx_e, _type_idx] = 1

        # get feed dict
        feed_dict[self.input_ids] = input_ids_np
        feed_dict[self.input_pos_ids] = input_pos_ids_np
        feed_dict[self.input_type_ids] = input_type_ids_np
        feed_dict[self.EO_label] = EO_label_np
        feed_dict[self.entity_type_label] = entity_type_label_np
        # feed_dict[self.predicate_label] = predicate_label_np
        # feed_dict[self.type_label] = type_label_np

        # for sketches: need u_lens_list, input_pos_ids_list
        if "lf" in example_batch[0]:
            sketch_ids_list = []
            sketch_entity_list = []
            sketch_predicate_list = []
            sketch_type_list = []
            sketch_num_list = []
            dict_ut2uid = {"prev_q":0, "prev_a": 2, "cur_q": 4}
            max_decoder_len = 2
            for _idx_ex, _example in enumerate(example_batch):
                u_lens = u_lens_list[_idx_ex]
                input_pos_ids = input_pos_ids_list[_idx_ex]

                # get token lens
                t_lens = []
                t_accu_lens = []
                _accu_len = 0
                _u_ptr = 0
                for u_len in u_lens:
                    u_token_pos_ids = input_pos_ids[_u_ptr:(_u_ptr+u_len)]
                    t_len = u_token_pos_ids[-1] - u_token_pos_ids[0] + 1
                    t_lens.append(t_len)
                    t_accu_lens.append(_accu_len)
                    _accu_len += t_len
                    _u_ptr += u_len

                sketch_ids = _example["lf"]["gold_sketch_ids"]  # dsl+1
                sketch_entity = []  # dsl
                sketch_predicate = []  # dsl
                sketch_type = []  # dsl
                sketch_num = []  # dsl
                invalid_ex = False
                if 0 < len(sketch_ids) <= max_seq_len - 2:
                    sketch_ids = [self.sos_id] + sketch_ids + [self.eos_id]
                    for _skt_t, _leaf in zip(_example["lf"]["gold_sketch"], _example["lf"]["gold_leaves_ids"] ):
                        sketch_entity.append(-1)  # index
                        sketch_predicate.append(self.pad_id)
                        sketch_type.append(self.pad_id)
                        sketch_num.append(-1)  # index
                        if _skt_t == "e":
                            _base_index = dict_ut2uid[_leaf[1]]
                            # _u_len = u_lens[_base_index]
                            _t_len = t_lens[_base_index]
                            _t_accu_len = t_accu_lens[_base_index]
                            if _leaf[0] < _t_len:  # (idx, ut)
                                sketch_entity[-1] = _t_accu_len + _leaf[0]
                            else:
                                invalid_ex = True  # label out of index
                                break
                        elif _skt_t == "r":
                            sketch_predicate[-1] = _leaf
                        elif _skt_t == "Type":
                            sketch_type[-1] = _leaf
                        elif _skt_t == "num_utterence":
                            _base_index = dict_ut2uid["cur_q"]
                            # _u_len = u_lens[_base_index]
                            _t_len = t_lens[_base_index]
                            _t_accu_len = t_accu_lens[_base_index]
                            if _leaf < _t_len:  # idx
                                sketch_num[-1] = _t_accu_len + _leaf
                            else:
                                invalid_ex = True # label out of index
                                break
                    sketch_entity.append(-1)  # index
                    sketch_predicate.append(self.pad_id)
                    sketch_type.append(self.pad_id)
                    sketch_num.append(-1)  # index
                else:  # the len of sketch exceed the max len
                    invalid_ex = True
                if invalid_ex:
                    sketch_ids = []
                    sketch_entity = []
                    sketch_predicate = []
                    sketch_type = []
                    sketch_num = []

                max_decoder_len = max(max_decoder_len, len(sketch_ids))

                sketch_ids_list.append(sketch_ids)  # dls, i.e., max_decoder_len
                sketch_entity_list.append(sketch_entity)  # dls, i.e., max_decoder_len-1
                sketch_predicate_list.append(sketch_predicate)  # dls, i.e., max_decoder_len-1
                sketch_type_list.append(sketch_type)  # dls, i.e., max_decoder_len-1
                sketch_num_list.append(sketch_num)  # dls, i.e., max_decoder_len-1

            sketch_ids_np = np.zeros([len(example_batch), max_decoder_len], self.cfg["intX"])
            sketch_entity_np = - np.ones([len(example_batch), max_decoder_len-1], self.cfg["intX"])
            sketch_predicate_np = np.zeros([len(example_batch), max_decoder_len-1], self.cfg["intX"])
            sketch_type_np = np.zeros([len(example_batch), max_decoder_len-1], self.cfg["intX"])
            sketch_num_np = - np.ones([len(example_batch), max_decoder_len-1], self.cfg["intX"])

            for _idx_ex in range(len(example_batch)):
                for _idx_t, _id in enumerate(sketch_ids_list[_idx_ex]):
                    sketch_ids_np[_idx_ex, _idx_t] = _id

                for _idx_t, _id in enumerate(sketch_entity_list[_idx_ex]):
                    sketch_entity_np[_idx_ex, _idx_t] = _id
                for _idx_t, _id in enumerate(sketch_predicate_list[_idx_ex]):
                    sketch_predicate_np[_idx_ex, _idx_t] = _id
                for _idx_t, _id in enumerate(sketch_type_list[_idx_ex]):
                    sketch_type_np[_idx_ex, _idx_t] = _id
                for _idx_t, _id in enumerate(sketch_num_list[_idx_ex]):
                    sketch_num_np[_idx_ex, _idx_t] = _id

            feed_dict[self.sketch_label] = sketch_ids_np
            feed_dict[self.sketch_entity] = sketch_entity_np
            feed_dict[self.sketch_predicate] = sketch_predicate_np
            feed_dict[self.sketch_type] = sketch_type_np
            feed_dict[self.sketch_num] = sketch_num_np

            if is_train_bool and self.cfg["use_qt_loss_gain"]:
                feed_dict[self.loss_gain_wrt_qt] = np.array(
                    [self.LOSS_GAIN_DICT[_example["question_type"]] for _example in example_batch], dtype="float32"
                )
            else:
                feed_dict[self.loss_gain_wrt_qt] = np.ones([len(example_batch)], dtype="float32")

        return feed_dict


def pos_ids_mask_gen(pos_ids, max_len):
    if len(pos_ids) <= max_len:
        return [1] * len(pos_ids)
    # delete from longest wordpiece
    # count
    dict_posid2freq = dict(collections.Counter(pos_ids))
    sorted_posid2freq = list(sorted(dict_posid2freq.items(), key=lambda e: e[0]))

    dict_freq2posids = {}
    for _posid, _freq in dict_posid2freq.items():
        if _freq not in dict_freq2posids:
            dict_freq2posids[_freq] = []
        dict_freq2posids[_freq].append(_posid)

    cur_freq = max(dict_freq2posids.keys())
    cur_len = len(pos_ids)
    dict_posid2redlen = {}
    while True:
        if len(dict_freq2posids) == 1 and list(dict_freq2posids.keys())[0] == 1:  # stop cond 1
            break

        red_posids = dict_freq2posids[cur_freq]
        red_len = len(red_posids)
        if red_len < cur_len - max_len:
            for _posid in red_posids:
                if _posid not in dict_posid2redlen:
                    dict_posid2redlen[_posid] = 0
                dict_posid2redlen[_posid] += 1

            if cur_freq-1 not in dict_freq2posids:
                dict_freq2posids[cur_freq-1] = []
            dict_freq2posids[cur_freq - 1].extend(red_posids)

            dict_freq2posids.pop(cur_freq)
            cur_freq -= 1
            cur_len -= red_len
        else:  # stop cond 2
            real_red_len = cur_len - max_len
            real_red_posids = red_posids[real_red_len:]
            for _posid in real_red_posids:
                if _posid not in dict_posid2redlen:
                    dict_posid2redlen[_posid] = 0
                dict_posid2redlen[_posid] += 1
            break

    res_mask = []
    for _posid, _freq in sorted_posid2freq:
        if _posid in dict_posid2redlen:
            red_len = dict_posid2redlen[_posid]
            assert red_len < _freq
            res_mask.extend(
                ([1] * (_freq-red_len)) + ([0] * red_len)
            )
        else:
            res_mask.extend([1]*_freq)
    return res_mask
