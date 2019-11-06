import tensorflow as tf
from peach.tf_nn.attn import compatibility_fn
from peach.tf_nn.general import get_shape_list, act_name2fn, mask_v3
from peach.tf_nn.nn import bn_dense_layer_v2
from peach.tf_nn.data.srl import generate_mask_based_on_lens, generate_seq_idxs


def logits_for_sketch_index(
        decoder_states, encoder_states,
        hn=None, wd=0., keep_prob=1.0, is_train=None,
        compress_mask=None,
        scope=None,
):
    compressing = not isinstance(compress_mask, type(None))
    hn = hn or get_shape_list(decoder_states)[-1]
    with tf.variable_scope(scope or "logits_for_sketch_index"):
        if compressing:
            new_decoder_states, _, rev_d = compress_seq_wrt_mask(decoder_states, compress_mask)
        else:
            new_decoder_states = decoder_states
            rev_d = None
        with tf.variable_scope("projection"):
            encoder_states_map = bn_dense_layer_v2(
                encoder_states, hn, True, 0., "encoder_states_map", "linear", False,
                wd, keep_prob, is_train
            )
            decoder_states_map = bn_dense_layer_v2(
                new_decoder_states, hn, True, 0., "decoder_states_map", "linear", False,
                wd, keep_prob, is_train
            )
        with tf.variable_scope("bi_linear"):
            bilinear_pre = bn_dense_layer_v2(
                decoder_states_map, hn, False, 0., "bilinear_map", "linear", False,
                wd, keep_prob, is_train
            )
            logits = tf.matmul(bilinear_pre, encoder_states_map, transpose_b=True)  # bs,dsl,esl

            if compressing:
                logits = decompress_seq_wrt_mask(logits, rev_d)

            return logits


def logits_for_sketch_prediction(
        decoder_states, cls_state, num_channel,
        hn=None, act_name="relu", wd=0., keep_prob=1.0, is_train=None,
        compress_mask=None,
        scope=None
):
    compressing = not isinstance(compress_mask, type(None))
    hn = hn or get_shape_list(decoder_states)[-1]
    with tf.variable_scope(scope or "logits_for_sketch_index"):
        if compressing:
            new_decoder_states, _, rev_d = compress_seq_wrt_mask(decoder_states, compress_mask)
        else:
            new_decoder_states = decoder_states
            rev_d = None
        map_part1 = bn_dense_layer_v2(
            new_decoder_states, hn, True, 0., "map_part1", "linear", False, wd, keep_prob, is_train
        )
        map_part2_pre = bn_dense_layer_v2(
            cls_state, hn, False, 0., "map_part2_pre", "linear", False, wd, keep_prob, is_train
        )
        map_part2 = tf.tile(tf.expand_dims(map_part2_pre, axis=1),
                            [1, get_shape_list(map_part1)[1], 1])
        map_res = act_name2fn(act_name)(map_part1 + map_part2)

        logits = bn_dense_layer_v2(
            map_res, num_channel, True, 0., "logits", "linear", False, wd, keep_prob, is_train
        )
        if compressing:
            logits = decompress_seq_wrt_mask(logits, rev_d)
        return logits


def compress_seq_wrt_mask(tensor_input, tensor_mask):

    bs, sl, hn = get_shape_list(tensor_input)

    seq_lens = tf.reduce_sum(tf.cast(tensor_mask, tf.int32), -1)  # sl
    max_len = tf.reduce_max(seq_lens)  # []
    new_mask = generate_mask_based_on_lens(seq_lens, max_len)

    # ======> to ensure every batch get same elem via padding
    pad_lens = max_len - seq_lens
    max_pad_len = tf.reduce_max(pad_lens)
    pad_mask = generate_mask_based_on_lens(pad_lens, max_pad_len)

    padded_tensor_mask = tf.concat([tensor_mask, pad_mask], axis=-1)  # bs,sl+max_pad_len
    # new coord
    bs_idxs = generate_seq_idxs(bs, sl+max_pad_len, transpose=True)  # bs,sl+max_pad_len
    sl_idxs = tf.concat(  # bs,sl+max_pad_len
        [
            generate_seq_idxs(bs, sl, transpose=False),  # bs,sl
            - tf.ones([bs, max_pad_len], tf.int32)  # bs, max_pad_len
        ], axis=-1
    )
    data_coord_map = tf.stack([bs_idxs, sl_idxs], axis=-1)  # bs,sl+max_pad_len,2

    padded_coord = tf.where(padded_tensor_mask)  # bs*max_len,2

    mapped_padded_coord_rsp = tf.gather_nd(data_coord_map, padded_coord)  # bs*max_len,2
    mapped_padded_coord = tf.reshape(mapped_padded_coord_rsp, [bs, max_len, 2])  # bs,max_len,2

    gathered_data = tf.gather_nd(tensor_input, mapped_padded_coord)  # bs,max_len,hn
    masked_gathered_data = mask_v3(gathered_data, new_mask, high_dim=True)

    reverse_dict = {
        "src_mask": tensor_mask,
        "tgt_mask": new_mask,
        "coord": mapped_padded_coord,  # bs,max_len,2
    }

    return masked_gathered_data, new_mask, reverse_dict


def decompress_seq_wrt_mask(tensor_input, reverse_dict):
    bs, tgt_len, hn = get_shape_list(tensor_input)
    src_len = get_shape_list(reverse_dict["src_mask"])[1]

    padded_tensor = tf.scatter_nd(reverse_dict["coord"], tensor_input, [bs, src_len+1, hn])
    out_tensor = padded_tensor[:,:-1]  # bs,src_len,hn

    masked_out_tensor = mask_v3(out_tensor, reverse_dict["src_mask"], high_dim=True)
    return masked_out_tensor


# if __name__ == '__main__':
#     # test the compress and decompress
#     data_list = [
#         [1, 0, 0, 0, 2, 0],
#         [3, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0],
#         [0, 0, 4, 0, 0, 0],
#         [0, 0, 0, 0, 5, 0],
#         [0, 3, 0, 5, 5, 0],
#     ]
#
#     raw_data = tf.convert_to_tensor(data_list, tf.int32)
#
#     data_ft = tf.tile(tf.expand_dims(tf.cast(raw_data, tf.float32), -1), [1, 1, 1])
#     data_mask = tf.cast(raw_data, tf.bool)
#
#     out_tensor, new_mask, rev_d = compress_seq_wrt_mask(data_ft, data_mask)
#     out_tensor_1 = out_tensor * 2
#     rev_tensor = decompress_seq_wrt_mask(out_tensor_1, rev_d)
#
#     sess = tf.Session()
#
#     print(sess.run(
#         {"out_tensor": out_tensor[...,0], "new_mask": new_mask, "coord": rev_d["coord"], "rev_tensor": rev_tensor[...,0]}
#     ))
#
#
#
#
#

