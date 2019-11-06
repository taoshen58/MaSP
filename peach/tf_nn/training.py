import tensorflow as tf
import re

from peach.tf_nn.general import  act_name2fn
from peach.tf_nn.nn import dropout, conv1d


def get_trainable_vars(scope, keys=tuple()):
    assert isinstance(keys, (tuple, list))
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    if len(keys) == 0:
        return trainable_vars
    else:
        regex_pattern = ".*{}.*".format(".*".join(keys))
        new_trainable_vars = []
        for var in trainable_vars:
            if re.match(regex_pattern, var.op.name):
                new_trainable_vars.append(var)
        return new_trainable_vars


# ============= Tasks Logits ==========

def snli_logits_sentence_encoding(s1_rep, s2_rep, afn, n_state, is_train, clf_dropout, highway=False):  # TODO: change this to my style (bn_dense_layer)
    out_rep = tf.concat([s1_rep, s2_rep, s1_rep - s2_rep, s1_rep * s2_rep], -1)
    act = act_name2fn(afn)
    h = act(conv1d(out_rep, 'c_fc', n_state, 1, train=is_train))

    if highway:
        trans = conv1d(h, 'c_trans', n_state, 1, train=is_train)
        gate = tf.nn.sigmoid(conv1d(h, 'c_gate', n_state, 1, train=is_train))
        h = gate * trans + (1-gate) * h

    h_dp = dropout(h, clf_dropout, is_train)
    return conv1d(h_dp, 'c_logits', 3, 1, train=is_train)


def qqp_logits_sentence_encoding(s1_rep, s2_rep, afn, n_state, is_train, clf_dropout, highway=False):   # TODO: change this to my style (bn_dense_layer)
    out_rep = tf.concat([tf.abs(s1_rep - s2_rep), s1_rep * s2_rep], -1)
    act = act_name2fn(afn)
    h = act(conv1d(out_rep, 'c_fc', n_state, 1, train=is_train))

    if highway:
        trans = conv1d(h, 'c_trans', n_state, 1, train=is_train)
        gate = tf.nn.sigmoid(conv1d(h, 'c_gate', n_state, 1, train=is_train))
        h = gate * trans + (1 - gate) * h

    h_dp = dropout(h, clf_dropout, is_train)
    return conv1d(h_dp, 'c_logits', 2, 1, train=is_train)
