"""
Note:
    1. All multi-head fn: the head dim always on the 2nd
"""

import tensorflow as tf
from peach.tf_nn.general import get_shape_list, act_name2fn, exp_mask_v3, mask_v3
from peach.tf_nn.nn import bn_dense_layer_v2, split_head, combine_head, bn_dense_layer_multi_head, dropout, \
    residual_connection, residual_connection_with_dense, masked_dense2sparse, masked_sparse2dense
from peach.tf_nn.utils.other import log_specific_params
import math
import logging


def cond_attn(
        pairwise_scores, featurewise_scores, value_features, from_mask, to_mask,
        attn_keep_prob=1., is_training=None,
        extra_pairwise_mask=None, name=None
):
    """

    :param pairwise_scores: [bs,[head],slf,slt]
    :param featurewise_scores:  [bs,[head],slt,hn]
    :param value_features:  [bs,[head],slt,hn]
    :param from_mask:
    :param to_mask:
    :param extra_pairwise_mask:
    :return:
    """
    with tf.name_scope(name or 'cond_attn'):
        # sanity check
        pairwise_shape = get_shape_list(pairwise_scores)
        featurewise_shape = get_shape_list(featurewise_scores)
        value_shape = get_shape_list(value_features)

        pairwise_ndim = len(pairwise_shape)
        featurewise_ndim = len(featurewise_shape)
        value_ndim = len(value_shape)

        assert featurewise_shape[-1] == value_shape[-1]
        assert pairwise_ndim in [3, 4] and pairwise_ndim == featurewise_ndim and featurewise_ndim == value_ndim

        multi_head = True if pairwise_ndim == 4 else False  # if the multi-head included

        cross_attn_mask = cross_attn_mask_generation(  # [bs,slf,slt]
            from_mask, to_mask, mutual=True
        )

        if multi_head:  # add the multi-head dim
            cross_attn_mask = tf.expand_dims(cross_attn_mask, 1)  # [bs,[1],slf,slt]

        if not isinstance(extra_pairwise_mask, type(None)):
            # the extra_pairwise_mask could be include the multi-head
            extra_pairwise_mask_shape = get_shape_list(extra_pairwise_mask)
            assert len(extra_pairwise_mask_shape) in [3, 4]

            assert multi_head or len(extra_pairwise_mask_shape) == 3  # if multi_head=False, shape must be 3-D

            if multi_head and len(extra_pairwise_mask_shape) == 3:
                extra_pairwise_mask = tf.expand_dims(cross_attn_mask, 1)  # [bs,[1],slf,slt]

            cross_attn_mask = tf.logical_and(cross_attn_mask, extra_pairwise_mask)  # [bs,[1],slf,slt]

        e_dot_logits = mask_v3(  # bs,head,sl1,sl2
            tf.exp(pairwise_scores), cross_attn_mask, multi_head=False, high_dim=False)  # the multi-head has been add

        e_multi_logits = mask_v3(
            tf.exp(featurewise_scores), to_mask, multi_head=multi_head, high_dim=True
        )

        with tf.name_scope("hybrid_attn"):
            # Z: softmax normalization term in attention probabilities calculation
            accum_z_deno = tf.matmul(e_dot_logits, e_multi_logits)  # num,bs,sl,dim
            accum_z_deno = tf.where(  # in case of NaN and Inf
                tf.greater(accum_z_deno, tf.zeros_like(accum_z_deno)),
                accum_z_deno,
                tf.ones_like(accum_z_deno)
            )
            # attention dropout
            e_dot_logits = dropout(e_dot_logits, math.sqrt(attn_keep_prob), is_training)
            e_multi_logits = dropout(e_multi_logits, math.sqrt(attn_keep_prob), is_training)
            # sum of exp(logits) \multiply attention target sequence
            rep_mul_score = value_features * e_multi_logits
            accum_rep_mul_score = tf.matmul(e_dot_logits, rep_mul_score)
            # calculate the final attention results
            attn_res = accum_rep_mul_score / accum_z_deno

        if multi_head:
            attn_res = combine_head(attn_res)  # [bs,slf,hd_num*hd_dim]

    return attn_res  # [bs,slf,hn/hd_num*hd_dim]


def softsel(
        attn_to_input, align_scores, attn_to_mask,  # {bs[,hd_num],slt,hn}, {bs[,hd_num],slf,slt[,hn]}, {bs,slt}
        mask_add_head_dim_for_scores=False,
        input_add_multi_head_dim=False,
        score_add_hn_dim=False,
        axis=-2, name=None):
    with tf.name_scope(name or 'softsel'):
        # ==== 1. attn to input =====
        if input_add_multi_head_dim:  # {bs[,hd_num],slt,hd_dim}
            attn_to_input = tf.expand_dims(attn_to_input, 1)
        attn_to_input = tf.expand_dims(attn_to_input, -3)  # {bs[,hd_num],1,slt,hd_dim}

        # 2. attn_to_mask
        attn_to_mask = tf.expand_dims(attn_to_mask, -2)  # {bs,1,slt}

        # ==== 2. align_scores =====
        if score_add_hn_dim:
            align_scores = tf.expand_dims(align_scores, -1)  # bs[,hd_num],slf,slt,(hn/1)

        masked_align_scores = exp_mask_v3(  # bs[,hd_num],slf,slt,(hn/1)
            align_scores, attn_to_mask, multi_head=mask_add_head_dim_for_scores, high_dim=True)

        attn_probs = tf.nn.softmax(masked_align_scores, axis)  # bs[,hd_num],slf,slt,(hn/1)
        attn_res = tf.reduce_sum(attn_probs * attn_to_input, axis=-2)  # bs[,hd_num],slf,(hn/1)
        return attn_res


def compatibility_fn(tensor_from, tensor_to, method='dot_product', scope=None, **kwargs):
    def _get_val_from_kwargs(key, default_val):
        if key in kwargs:
            return kwargs[key]
        else:
            return default_val

    with tf.variable_scope(scope or 'compatibility_fn.{}'.format(method)):
        shape_from = get_shape_list(tensor_from)
        ndim_from = len(shape_from)
        shape_to = get_shape_list(tensor_to)
        ndim_to = len(shape_to)

        assert ndim_from == ndim_to or ndim_from+1 == ndim_to
        need_extra_dim = ndim_from+1 == ndim_to

        if need_extra_dim:
            tensor_from = tf.expand_dims(tensor_from, -2)
            shape_from = get_shape_list(tensor_from)

        slf, slt = shape_from[-2], shape_to[-2]

        # hparams parsing
        hn = _get_val_from_kwargs('hn', shape_to[-1])
        wd = _get_val_from_kwargs('wd', 0.)
        keep_prob = _get_val_from_kwargs('keep_prob', 1.)
        is_training = _get_val_from_kwargs('is_training', None)
        activation = _get_val_from_kwargs('activation', 'relu')
        head_num = _get_val_from_kwargs('head_num', 12)

        seq_dim_to_remove = -3
        if method == 'dot_product':
            align_scores = tf.matmul(tensor_from, tensor_to, transpose_b=True)  # [bs,slf,hn]*[bs,slt,hn]=>bs,slf,slt
            align_scores = tf.expand_dims(align_scores, -1)  # [bs,slf,slt,1]
        elif method == 'additive':
            tensor_from_branch = bn_dense_layer_v2(
                tensor_from, hn, False, 0., 'tensor_from_branch', 'linear', False,
                wd, keep_prob, is_training
            )
            tensor_to_branch = bn_dense_layer_v2(
                tensor_to, hn, True, 0., 'tensor_to_branch', 'linear', False,
                wd, keep_prob, is_training
            )
            align_scores_pre = act_name2fn(activation)(tf.add(  # [bs,slf,slt,hn]
                tf.expand_dims(tensor_from_branch, -2),  # [bs,slf,1,hn]
                tf.expand_dims(tensor_to_branch, -3)  # [bs,1,slt,hn]
            ))
            align_scores = bn_dense_layer_v2(  # [bs,slf,slt,1]
                align_scores_pre, 1, True, 0., 'align_scores', 'linear', False,
                wd, keep_prob, is_training
            )
        elif method == 'multi_dim':
            logging.warning("No simplified multi-dim technique used in this function!")
            tensor_from_branch = bn_dense_layer_v2(
                tensor_from, hn, False, 0., 'tensor_from_branch', 'linear', False,
                wd, keep_prob, is_training
            )
            tensor_to_branch = bn_dense_layer_v2(
                tensor_to, hn, True, 0., 'tensor_to_branch', 'linear', False,
                wd, keep_prob, is_training
            )
            align_scores_pre = act_name2fn(activation)(tf.add(  # [bs,slf,slt,hn]
                tf.expand_dims(tensor_from_branch, -2),  # [bs,slf,1,hn]
                tf.expand_dims(tensor_to_branch, -3)  # bs,1,slt,hn
            ))
            align_scores = bn_dense_layer_v2(  # [bs,slf,slt,hn]
                align_scores_pre, hn, True, 0., 'align_score', 'linear', False,
                wd, keep_prob, is_training
            )
        elif method == 'multi_head':
            seq_dim_to_remove = -2  # !!! because multi-head dim is on 2nd dim
            assert hn % head_num == 0
            head_dim = hn // head_num

            q_heads = bn_dense_layer_v2(
                tensor_from, head_dim, True, 0., 'q_heads',
                'linear', False, wd, keep_prob, is_training, dup_num=head_num
            )
            k_heads = bn_dense_layer_v2(
                tensor_to, head_dim, True, 0., 'k_heads',
                'linear', False, wd, keep_prob, is_training, dup_num=head_num
            )
            q_heads = split_head(q_heads, head_num)  # bs,hd_num,slf,hd_dim
            k_heads = split_head(k_heads, head_num)  # bs,hd_num,slt,hd_dim

            # alignment score
            align_scores = tf.matmul(q_heads, k_heads, transpose_b=True)  # [bs,hd_num,slf,slt]
            align_scores = align_scores / math.sqrt(1.*head_dim)  # [bs,hd_num,slf,slt]
        elif method in ['multi_head_bilinear', 'multi_head_bilinear_shared', 'multi_head_only', 'multi_head_linear']:
            seq_dim_to_remove = -2  # !!! because multi-head dim is on 2nd dim
            assert hn % head_num == 0
            head_dim = hn // head_num

            q_heads = bn_dense_layer_v2(
                tensor_from, head_dim, True, 0., 'q_heads',
                kwargs.get("activation") or activation,
                False, wd, keep_prob, is_training, dup_num=head_num
            )
            k_heads = bn_dense_layer_v2(
                tensor_to, head_dim, True, 0., 'k_heads',
                kwargs.get("activation") or activation,
                False, wd, keep_prob, is_training, dup_num=head_num
            )
            q_heads = split_head(q_heads, head_num)  # bs,hd_num,slf,hd_dim
            k_heads = split_head(k_heads, head_num)  # bs,hd_num,slt,hd_dim

            # alignment score: using biliear rather than dot product
            # align_scores = tf.matmul(q_heads, k_heads, transpose_b=True)  # [bs,hd_num,slf,slt]
            # align_scores = align_scores / math.sqrt(1. * head_dim)  # [bs,hd_num,slf,slt]
            with tf.variable_scope("bilinear"):
                if method == "multi_head_bilinear":
                    k_heads_map = bn_dense_layer_multi_head(
                        k_heads, head_dim, False, 0., 'k_heads_map', 'linear', False, wd, keep_prob, is_training)
                elif method == "multi_head_bilinear_shared":
                    k_heads_map = bn_dense_layer_v2(
                        k_heads, head_dim, False, 0., 'k_heads_map', 'linear', False, wd, keep_prob, is_training)
                elif method == "multi_head_only":
                    pass
                elif method == "multi_head_linear":
                    k_heads_map = bn_dense_layer_v2(
                        k_heads, head_dim, False, 0., 'k_heads_map', 'linear', False, wd, keep_prob, is_training)
                    q_heads_map = bn_dense_layer_v2(
                        q_heads, head_dim, False, 0., 'q_heads_map', 'linear', False, wd, keep_prob, is_training)
                else:
                    raise AttributeError
                align_scores = tf.matmul(q_heads, k_heads, transpose_b=True)

                log_specific_params()


        elif method == 'multi_dim_head':
            assert hn % head_num == 0
            head_dim = hn // head_num

            q_heads = bn_dense_layer_v2(
                tensor_from, head_dim, True, 0., 'q_heads',
                'linear', False, wd, keep_prob, is_training, dup_num=head_num
            )
            k_heads = bn_dense_layer_v2(
                tensor_to, head_dim, True, 0., 'k_heads',
                'linear', False, wd, keep_prob, is_training, dup_num=head_num
            )
            q_heads = split_head(q_heads, head_num)  # bs,hd_num,slf,hd_dim
            k_heads = split_head(k_heads, head_num)  # bs,hd_num,slt,hd_dim

            # MLP
            q_heads_branch = bn_dense_layer_multi_head(
                q_heads, head_dim, False, 0., 'q_heads_branch', 'linear', False,
                wd, keep_prob, is_training
            )
            k_heads_branch = bn_dense_layer_multi_head(
                k_heads, head_dim, True, 0., 'k_heads_branch', 'linear', False,
                wd, keep_prob, is_training
            )
            align_scores_pre = act_name2fn(activation)(tf.add(  # [bs,head,slf,slt,dim]
                tf.expand_dims(q_heads_branch, -2),  # [bs,head,slf,1,dim]
                tf.expand_dims(k_heads_branch, -3)  # bs,head,1,slt,dim
            ))
            align_scores_heads = bn_dense_layer_multi_head(  # [bs,hd_num,slf,slt,hd_dim]
                align_scores_pre, head_dim, True, 0., 'align_scores_heads', 'linear', False,
                wd, keep_prob, is_training
            )
            align_scores = align_scores_heads  # [bs,hd_num,slf,slt,hd_dim]
        elif method == 'bilinear':
            raise NotImplementedError
        else:
            raise AttributeError

        if need_extra_dim:
            align_scores = tf.squeeze(align_scores, [seq_dim_to_remove])  #

        return align_scores


def attn_post_proc(attn_res, inter_hn=None, wd=0., keep_prob=1., residual_keep_prob=1.,
                   is_train=None, activation='relu', sparse_opt=False,
                   scope=None, **kwargs):
    with tf.variable_scope(scope or "attn_res"):
        assert "mask" in kwargs
        if sparse_opt:
            x1, reverse_spec = masked_dense2sparse(attn_res, kwargs.get("mask"))
        else:
            x1 = attn_res

        y = bn_dense_layer_v2(
            x1, get_shape_list(attn_res)[-1], True, 0., "dense_layer", "linear", False,
            wd, keep_prob, is_train

        )
        x2 = residual_connection(x1, y, is_train, residual_keep_prob, "res_con")

        res = residual_connection_with_dense(
            x2, inter_hn or 4*get_shape_list(attn_res)[-1], True, 0., "residual_connection_with_dense",
            activation, False, wd, keep_prob, is_train, residual_keep_prob
        )
        if sparse_opt:
            res = masked_sparse2dense(res, reverse_spec)
        return res


# source2token self attention
def s2t_self_attn(
        tensor_input, tensor_mask, deep_act=None, method='multi_dim',
        wd=0., keep_prob=1., is_training=None,
        scope=None, **kwargs
):
    use_deep = isinstance(deep_act, str)  # use Two layers or Single layer for the alignment score
    with tf.variable_scope(scope or 's2t_self_attn_{}'.format(method)):
        tensor_shape = get_shape_list(tensor_input)
        hn = tensor_shape[-1]  # hidden state number

        if method == 'additive':
            align_scores = bn_dense_layer_v2(  # bs,sl,hn/1
                tensor_input, hn if use_deep else 1, True, 0., 'align_score_1', 'linear', False,
                wd, keep_prob, is_training
            )
            if use_deep:
                align_scores = bn_dense_layer_v2(  # bs,sl,1
                    act_name2fn(deep_act)(align_scores), 1, True, 0., 'align_score_2', 'linear', False,
                    wd, keep_prob, is_training
                )
        elif method == 'multi_dim':
            align_scores = bn_dense_layer_v2(  # bs,sl,hn
                tensor_input, hn, False, 0., 'align_score_1', 'linear', False,
                wd, keep_prob, is_training
            )
            if use_deep:
                align_scores = bn_dense_layer_v2(  # bs,sl,hn
                    act_name2fn(deep_act)(align_scores), hn, True, 0., 'align_score_2', 'linear', False,
                    wd, keep_prob, is_training
                )
        elif method == 'multi_dim_head':
            get_shape_list(tensor_input, expected_rank=3)  # the input should be rank-3
            assert 'head_num' in kwargs and isinstance(kwargs['head_num'], int)
            head_num = kwargs['head_num']
            assert hn % head_num == 0
            head_dim = hn // head_num

            tensor_input_heads = split_head(tensor_input, head_num)  # [bs,hd,sl,hd_dim]

            align_scores_heads = bn_dense_layer_multi_head(  # [bs,hd,sl,hd_dim]
                tensor_input_heads, head_dim, True, 0., 'align_scores_heads_1', 'linear', False,
                wd, keep_prob, is_training
            )
            if use_deep:
                align_scores_heads = bn_dense_layer_multi_head(  # [bs,hd,sl,hd_dim]
                    act_name2fn(deep_act)(align_scores_heads), head_dim,
                    True, 0., 'align_scores_heads_2', 'linear', False,
                    wd, keep_prob, is_training
                )
            align_scores = combine_head(align_scores_heads)  # [bs,sl,dim]
        else:
            raise AttributeError

        # attention procedure align_scores [bs,sl,1/dim]
        align_scores_masked = exp_mask_v3(align_scores, tensor_mask, multi_head=False, high_dim=True)  # bs,sl,hn
        attn_prob = tf.nn.softmax(align_scores_masked, axis=-2)  # bs,sl,hn

        if 'attn_keep_prob' in kwargs and isinstance(kwargs['attn_keep_prob'], float):
            attn_prob = dropout(attn_prob, kwargs['attn_keep_prob'], is_training)  # bs,sl,hn

        attn_res = tf.reduce_sum(  # [bs,sl,hn] -> [bs,dim]
            mask_v3(attn_prob*tensor_input, tensor_mask, high_dim=True), axis=-2
        )

        return attn_res  # [bs,hn]


# todo attention mechanism


# ============= t2t self-attention =================
# todo: token2token self-attention with direction encoded


def direct_mask_generation(rep_mask, direct, attn_self, name=None):
    assert direct in ["forward", "backward"]
    with tf.name_scope(name or 'direct_mask_generation'):
        rep_shape = get_shape_list(rep_mask, 2)
        bs, sl = rep_shape
        # regular mask
        rep_mask_epd1 = tf.expand_dims(rep_mask, 1)  # bs,1,sl
        rep_mask_epd2 = tf.expand_dims(rep_mask, 2)  # bs,sl,1
        rep_mask_mat = tf.logical_and(rep_mask_epd1, rep_mask_epd2)  # bs,sl,sl

        # position mask
        sl_indices = tf.range(sl, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)

        comp_func = tf.greater_equal if attn_self else tf.greater
        if direct == "forward":
            direct_mask = comp_func(sl_row, sl_col)  # sl,sl
        elif direct == "backward":
            direct_mask = comp_func(sl_col, sl_row)
        else:
            raise AttributeError
        direct_mask = tf.tile(tf.expand_dims(direct_mask, 0), [bs, 1, 1])

        return tf.logical_and(rep_mask_mat, direct_mask)


def mask_generation(rep_mask, head_num, use_direction, attn_self, name=None):  # this mask is for self-attention
    with tf.name_scope(name or 'mask_generation'):
        rep_shape = get_shape_list(rep_mask, 2)
        bs, sl = rep_shape
        # regular mask
        rep_mask_epd1 = tf.expand_dims(rep_mask, 1)  # bs,1,sl
        rep_mask_epd2 = tf.expand_dims(rep_mask, 2)  # bs,sl,1
        rep_mask_mat = tf.logical_and(rep_mask_epd1, rep_mask_epd2)  # bs,sl,sl

        # position mask
        sl_indices = tf.range(sl, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)

        if use_direction:
            comp_func = tf.greater_equal if attn_self else tf.greater
            fw_mask = comp_func(sl_row, sl_col)  # sl,sl
            bw_mask = comp_func(sl_col, sl_row)  # sl,sl
            direct_mask = tf.stack([fw_mask, bw_mask], 0)  # 2,sl,sl
            direct_mask = tf.reshape(  # num,sl,sl
                tf.tile(tf.expand_dims(direct_mask, 1), [1, int(head_num / 2), 1, 1]),  # 2,4,sl,sl
                [head_num, sl, sl])
        else:
            if not attn_self:
                direct_mask = tf.tile(tf.expand_dims(tf.not_equal(sl_row, sl_col), 0), [head_num, 1, 1])  # n,sl,sl
            else:
                raise(ValueError, "A attention overself must be avoided without fw/bw information")

        final_mask = tf.logical_and(  # bs,num,sl,sl
            tf.expand_dims(rep_mask_mat, 1),
            tf.expand_dims(direct_mask, 0))
        return final_mask


def mask_ft_generation(rep_mask, head_num, use_direction, attn_self):
    return tf.cast(mask_generation(rep_mask, head_num, use_direction, attn_self), tf.float32)


def cross_attn_mask_generation(from_mask, to_mask, mutual=True, head_num=None, name=None):
    """

    :param from_mask: 2-D Tensor, [bs,slf]
    :param to_mask: 2-D Tensor, [bs,slt]
    :param mutual:
    :param head_num
    :param name:
    :return: 3D Tensor
    """
    with tf.name_scope(name or 'attention_mask_generation'):
        bs, slf = get_shape_list(from_mask, 2)[:2]
        slt = get_shape_list(to_mask, 2)[1]

        if mutual:
            res_mask = tf.cast(  # [bs,slf,slt]
                tf.expand_dims(tf.cast(from_mask, tf.int32), 2) * tf.expand_dims(tf.cast(to_mask, tf.int32), 1),
                tf.bool
            )
        else:
            res_mask = tf.tile(tf.expand_dims(to_mask, 1), [1, slf, 1])  # [bs,slt] -> [bs,slf,slt]

        if isinstance(head_num, int):
            res_mask = tf.expand_dims(res_mask, 1)
            tile_multiples = [1] * len(get_shape_list(res_mask))
            tile_multiples[1] = head_num
            res_mask = tf.tile(res_mask, tile_multiples)

        return res_mask


# ======================================================================
# ============================ Lagacy Methods ===========================
def compatibility_fn_lacacy(  # did not support arbitrary dim
        tensor_from, tensor_to, method='dot_product', scope=None, **kwargs):

    def _get_val_from_kwargs(key, default_val):
        if key in kwargs:
            return kwargs[key]
        else:
            return default_val

    with tf.variable_scope(scope or 'compatibility_fn.{}'.format(method)):
        shape_from = get_shape_list(tensor_from)
        ndim_from = len(shape_from)
        shape_to = get_shape_list(tensor_to)
        ndim_to = len(shape_to)

        assert (ndim_from == 2 or ndim_from == 3) and ndim_to == 3

        if ndim_from == 2:
            tensor_from = tf.expand_dims(tensor_from, 1)
            shape_from = get_shape_list(tensor_from)

        slf, slt = shape_from[1], shape_to[1]

        # hparams parsing
        hn = _get_val_from_kwargs('hn', shape_to[-1])
        wd = _get_val_from_kwargs('wd', 0.)
        keep_prob = _get_val_from_kwargs('keep_prob', 1.)
        is_training = _get_val_from_kwargs('is_training', None)
        activation = _get_val_from_kwargs('activation', 'relu')
        head_num = _get_val_from_kwargs('head_num', 12)

        seq_dim_to_remove = 1
        if method == 'dot_product':
            align_scores = tf.matmul(tensor_from, tensor_to, transpose_b=True)  # [bs,slf,hn]*[bs,slt,hn]=>bs,slf,slt
            align_scores = tf.expand_dims(align_scores, -1)  # [bs,slf,slt,1]
        elif method == 'additive':
            tensor_from_branch = bn_dense_layer_v2(
                tensor_from, hn, False, 0., 'tensor_from_branch', 'linear', False,
                wd, keep_prob, is_training
            )
            tensor_to_branch = bn_dense_layer_v2(
                tensor_to, hn, True, 0., 'tensor_to_branch', 'linear', False,
                wd, keep_prob, is_training
            )
            align_scores_pre = act_name2fn(activation)(tf.add(  # [bs,slf,slt,hn]
                tf.expand_dims(tensor_from_branch, 2),  # [bs,slf,1,hn]
                tf.expand_dims(tensor_to_branch, 1)  # [bs,1,slt,hn]
            ))
            align_scores = bn_dense_layer_v2(  # [bs,slf,slt,1]
                align_scores_pre, 1, True, 0., 'align_scores', 'linear', False,
                wd, keep_prob, is_training
            )
        elif method == 'multi_dim':
            logging.warning("No simplified multi-dim technique used in this function!")
            tensor_from_branch = bn_dense_layer_v2(
                tensor_from, hn, False, 0., 'tensor_from_branch', 'linear', False,
                wd, keep_prob, is_training
            )
            tensor_to_branch = bn_dense_layer_v2(
                tensor_to, hn, True, 0., 'tensor_to_branch', 'linear', False,
                wd, keep_prob, is_training
            )
            align_scores_pre = act_name2fn(activation)(tf.add(  # [bs,slf,slt,hn]
                tf.expand_dims(tensor_from_branch, 2),  # [bs,slf,1,hn]
                tf.expand_dims(tensor_to_branch, 1)  # bs,1,slt,hn
            ))
            align_scores = bn_dense_layer_v2(
                align_scores_pre, hn, True, 0., 'align_score', 'linear', False,
                wd, keep_prob, is_training
            )
        elif method == 'multi_head':
            seq_dim_to_remove = 2  # !!! because multi-head dim is on 2nd dim
            assert hn % head_num == 0
            head_dim = hn // head_num

            q_heads = bn_dense_layer_v2(
                tensor_from, head_dim, True, 0., 'q_heads',
                'linear', False, wd, keep_prob, is_training, dup_num=head_num
            )
            k_heads = bn_dense_layer_v2(
                tensor_to, head_dim, True, 0., 'k_heads',
                'linear', False, wd, keep_prob, is_training, dup_num=head_num
            )
            q_heads = split_head(q_heads, head_num)  # bs,hd_num,slf,hd_dim
            k_heads = split_head(k_heads, head_num)  # bs,hd_num,slt,hd_dim

            # alignment score
            align_scores = tf.matmul(q_heads, k_heads, transpose_b=True)  # [bs,hd_num,slf,slt]
            align_scores = align_scores / math.sqrt(1.*head_dim)  # [bs,hd_num,slf,slt]
        elif method == 'multi_dim_head':
            seq_dim_to_remove = 2  # !!! because multi-head dim is on 2nd dim
            assert hn % head_num == 0
            head_dim = hn // head_num

            q_heads = bn_dense_layer_v2(
                tensor_from, head_dim, True, 0., 'q_heads',
                'linear', False, wd, keep_prob, is_training, dup_num=head_num
            )
            k_heads = bn_dense_layer_v2(
                tensor_to, head_dim, True, 0., 'k_heads',
                'linear', False, wd, keep_prob, is_training, dup_num=head_num
            )
            q_heads = split_head(q_heads, head_num)  # bs,hd_num,slf,hd_dim
            k_heads = split_head(k_heads, head_num)  # bs,hd_num,slt,hd_dim

            # MLP
            q_heads_branch = bn_dense_layer_multi_head(
                q_heads, head_dim, False, 0., 'q_heads_branch', 'linear', False,
                wd, keep_prob, is_training
            )
            k_heads_branch = bn_dense_layer_multi_head(
                k_heads, head_dim, True, 0., 'k_heads_branch', 'linear', False,
                wd, keep_prob, is_training
            )
            align_scores_pre = act_name2fn(activation)(tf.add(  # [bs,head,slf,slt,dim]
                tf.expand_dims(q_heads_branch, 3),  # [bs,head,slf,1,dim]
                tf.expand_dims(k_heads_branch, 2)  # bs,head,1,slt,dim
            ))
            align_scores_heads = bn_dense_layer_multi_head(  # [bs,hd_num,slf,slt,hd_dim]
                align_scores_pre, head_dim, True, 0., 'align_scores_heads', 'linear', False,
                wd, keep_prob, is_training
            )
            align_scores = align_scores_heads  # [bs,hd_num,slf,slt,hd_dim]
            # align_scores = combine_head(align_scores_heads)
        elif method == 'bilinear':
            raise NotImplementedError
        else:
            raise AttributeError

        if ndim_from == 2:
            align_scores = tf.squeeze(align_scores, [seq_dim_to_remove])  #

        return align_scores
