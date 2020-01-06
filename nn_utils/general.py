import tensorflow as tf
from peach.tf_nn.general import get_shape_list
from peach.tf_nn.data.srl import generate_mask_based_on_lens


def get_key_indices(tensor_input, special_token_list):
    # tensor_input 2
    get_shape_list(tensor_input, 2)
    out_indices_list = []
    for sp_token in special_token_list:
        out_indices_list.append(
            tf.cast(
                tf.argmax(tf.cast(tf.equal(tensor_input, sp_token), tf.int32), 1),
                tf.int32)
        )
    return out_indices_list


def get_slice(tensor_input, start_idxs, end_idxs):
    # 1. the size of 1st dim of tensor_input, start_idxs, end_idxs must be equal
    # 2. the idxs is the 2nd dim of tensor input
    # 3. output: 1. a output tensor, 2. a mask
    tensor_shape = get_shape_list(tensor_input)
    bs = tensor_shape[0]
    sl = tensor_shape[1]
    extra_dims = tensor_shape[2:] if len(tensor_shape) > 2 else []
    lens = end_idxs - start_idxs - 1
    max_len = tf.reduce_max(lens)

    # target bool indicator
    indices_input = tf.tile(tf.expand_dims(tf.range(sl, dtype=tf.int32), 0), [bs, 1])  # bs, sl
    indices_new = indices_input - tf.expand_dims(start_idxs, 1) - 1  # bs, sl
    tgt_bool_indicator = tf.logical_and(
        tf.greater(indices_input, tf.expand_dims(start_idxs, 1)),
        tf.less(indices_input, tf.expand_dims(end_idxs, 1)),
    )

    coord_in_input = tf.where(tgt_bool_indicator)  # [n_true, 2]
    two_d_indices_new = tf.stack(  # bs,sl,2
        values=[
            tf.tile(tf.expand_dims(tf.range(bs, dtype=tf.int32), 1), [1, sl]),
            indices_new,
        ], axis=-1
    )

    coord_in_output = tf.gather_nd(two_d_indices_new, coord_in_input)  # [n_true, 2]
    gathered_tensor_input = tf.gather_nd(tensor_input, coord_in_input)  # [n_true]+extra_dims

    tensor_output = tf.scatter_nd(coord_in_output, gathered_tensor_input, [bs, max_len] + extra_dims)
    mask_output = generate_mask_based_on_lens(lens, max_len)
    return tensor_output, mask_output


