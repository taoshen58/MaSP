import tensorflow as tf
from peach.tf_nn.general import mask_v3, get_shape_list


def get_word_level_split(params, input_pos_ids, wordpiece_idx, input_mask, sll, pl):
    # bs,sl,pl
    bs, sl = get_shape_list(input_pos_ids)

    higher_dim = len(get_shape_list(params)) > 2
    extra_dims = get_shape_list(params)[2:] if higher_dim else []

    # tf.tile(tf.expand_dims(tf.expand_dims(tf.range(bs), 1), 2), [1, sll, pl])
    bs_idxs = tf.tile(tf.expand_dims(tf.range(bs), 1), [1, sl])

    data_coord = tf.stack([bs_idxs, input_pos_ids, wordpiece_idx], -1)  # [bs, sl, 3]
    # mask input_pos_ids and wordpiece_idx for -1
    mask_reversed_int = tf.cast(tf.logical_not(input_mask), tf.int32)
    data_coord = mask_v3(data_coord, input_mask, high_dim=True) + tf.stack(
        [
            mask_reversed_int * bs,
            mask_reversed_int * sll,
            mask_reversed_int * pl,
        ], axis=-1
    )

    # params's dtype check
    is_bool = (params.dtype == tf.bool)
    outputs = tf.scatter_nd(
        indices=data_coord,  # [bs, sl, 3]
        updates=params if not is_bool else tf.cast(params, tf.int32),  # [bs,sl]
        shape=[bs+1, sll+1, pl+1] + extra_dims
    )
    if is_bool:
        outputs = tf.cast(outputs, tf.bool)

    outputs = outputs[:-1, :-1, :-1]
    return outputs


def generate_label_mask(input_pos_ids, input_mask, sll):
    input_pos_ids = mask_v3(input_pos_ids+1, input_mask)

    bs, sl = get_shape_list(input_pos_ids)

    sll_idxs = tf.tile(tf.expand_dims(tf.range(sll, dtype=tf.int32), 0), [bs, 1])  # bs,sl
    max_idxs = tf.reduce_max(input_pos_ids, axis=-1, keepdims=True)  # [bs,1]

    return tf.less(sll_idxs, max_idxs)


# transform the pos_ids to
def transform_pos_ids_to_wordpiece_idx(input_pos_ids, input_mask, sll):
    # 0 0 1 1 1 2 2 2 2 3 3 0 0 0 0 0  # bs,sl
    #
    bs, sl = get_shape_list(input_pos_ids)
    diff_pos = mask_v3(  # bs,sl
        input_pos_ids - tf.concat([tf.zeros([bs, 1], dtype=tf.int32), input_pos_ids[:, :-1]], axis=1),
        input_mask
    )

    sl_idxs = tf.tile(tf.expand_dims(tf.range(sl, dtype=tf.int32), 0), [bs, 1])  # bs,sl
    word_start_index = diff_pos * sl_idxs  # bs, sl
    # remove all 0 value
    slx_s = tf.reduce_sum(diff_pos, axis=-1)  # the number of non-zero for each example
    slx = tf.reduce_max(slx_s)  #
    sly_s = slx - slx_s  # the number of non-zero for padding
    sly = tf.reduce_max(sly_s)  #
    padding_seq = tf.cast(generate_mask_based_on_lens(sly_s, sly), tf.int32)
    valid_data_mask = generate_mask_based_on_lens(slx_s, slx)  # bs, slx

    padded_word_start_index = tf.concat([word_start_index, padding_seq], axis=-1)  # bs,sl+sly

    data_coord = tf.reshape(  # bs, slx
        tf.where(tf.cast(padded_word_start_index, tf.bool)),  # bs*slx,2
        [bs, slx, 2]
    )

    word_start = tf.concat(  # bs, sll
        [
            tf.zeros([bs, 1], dtype=tf.int32),
            mask_v3(tf.gather_nd(padded_word_start_index, data_coord), valid_data_mask),  # bs,slx
            tf.zeros([bs, sll - slx - 1], dtype=tf.int32)
        ],
        axis=1
    )

    bs_idxs = generate_seq_idxs(bs, sl, transpose=True)  # bs,sl
    base_coord = tf.stack([bs_idxs, input_pos_ids], axis=-1)  # bs,sl,2
    base_value = tf.gather_nd(word_start, base_coord)  # bs,sl

    # finally
    outputs = mask_v3(sl_idxs - base_value, input_mask)  # bs,sl
    return outputs


def generate_seq_idxs(bs, max_len, transpose=False):
    if transpose:
        return tf.tile(tf.expand_dims(tf.range(bs, dtype=tf.int32), 1), [1, max_len])
    else:
        return tf.tile(tf.expand_dims(tf.range(max_len, dtype=tf.int32), 0), [bs, 1])


def generate_mask_based_on_lens(lens, max_len):
    bs = get_shape_list(lens)[0]
    seq_idxs = generate_seq_idxs(bs, max_len)
    mask = tf.less(seq_idxs, tf.expand_dims(lens, axis=-1))
    return mask


def mask_matrix_to_coordinate(mask_mat, name=None):
    with tf.name_scope(name or "mask_matrix_to_coordinate"):
        bs, sll = get_shape_list(mask_mat, expected_rank=2)

        # lens
        real_lens = tf.reduce_sum(tf.cast(mask_mat, tf.int32), axis=-1)  # bs
        max_real_len = tf.reduce_max(real_lens, axis=0)  # []
        pad_lens = max_real_len - real_lens
        max_pad_len = tf.reduce_max(pad_lens, axis=0)

        # mask generation
        pad_mask_mat = generate_mask_based_on_lens(pad_lens, max_pad_len)
        coord_mask = generate_mask_based_on_lens(real_lens, max_real_len)

        # coord generation
        padded_mask_mat = tf.concat([mask_mat, pad_mask_mat], axis=-1)

        flat_coords = tf.where(padded_mask_mat)  # [bs*max_real_len,2]
        coords = tf.reshape(flat_coords, [bs, max_real_len, 2])  # [bs,max_real_len]
        coords = mask_v3(coords, coord_mask, high_dim=True)
        return coords, coord_mask


def top_k_to_coordinate(top_k_vec, prob_tensor=None, logits=None, dim=None, name=None):
    if isinstance(prob_tensor, type(None)):
        prob_tensor = tf.nn.softmax(logits, axis=-1)[..., dim]

    with tf.name_scope(name or "top_k_to_coordinate"):
        bs, sll = get_shape_list(prob_tensor, expected_rank=2)
        sorted_tensor = tf.contrib.framework.sort(prob_tensor, axis=-1, direction='DESCENDING')  # bs,sll

        padded_sorted_tensor = tf.concat([sorted_tensor, -tf.ones([bs, 1], sorted_tensor.dtype)], axis=-1)  # [bs,sll+1]
        k_th_scores_indices = tf.stack(  # [bs,2]
            [
                tf.range(bs, dtype=tf.int32),
                top_k_vec,
            ], axis=-1
        )
        k_th_scores = tf.expand_dims(tf.gather_nd(padded_sorted_tensor, k_th_scores_indices), axis=-1)  # [bs,1]

        mask_mat = tf.greater(prob_tensor, k_th_scores)  # [bs,sll]
        return mask_matrix_to_coordinate(mask_mat)


def compress_2nd_dim_to_batch(input_tensor, num_vec, name=None): # [bs,sd,...] -> [nbs,...]
    with tf.name_scope(name or "compress_2nd_dim_to_batch"):
        bs, sd = get_shape_list(input_tensor)[:2]
        num_mask = generate_mask_based_on_lens(num_vec, sd)  # [bs,sd]
        coords = tf.where(num_mask)  # [nbs,2]
        reverse_spec = {
            "org_coords": coords,
            "org_input_mask": num_mask,
        }
        out_tensor = tf.gather_nd(input_tensor, coords)  # [nbs,...]
        out_tensor = tf.expand_dims(out_tensor, 1)
        return out_tensor, reverse_spec


def decompress_2nd_dim_from_batch(input_tensor, reverse_spec, name=None):  # [nbs, 1,...] -> [bs,2d,...]
    with tf.name_scope(name or "decompress_2nd_dim_from_batch"):
        input_tensor_squeeze = tf.squeeze(input_tensor, 1)
        remain_shape = get_shape_list(input_tensor_squeeze)[1:]
        org_coords = reverse_spec["org_coords"]
        org_input_mask = reverse_spec["org_input_mask"]
        bs, sd = get_shape_list(org_input_mask)
        return tf.scatter_nd(org_coords, input_tensor_squeeze, [tf.to_int64(elem) for elem in [bs, sd]+remain_shape])


def extend_batch_for_2nd_dim_compression(input_tensor, reverse_spec=None, num_vec=None, name=None):  # [bs,...] -> [nbs,...]
    with tf.name_scope(name or "extend_batch_for_2nd_dim_compression"):

        if reverse_spec is not None:
            org_input_mask = reverse_spec["org_input_mask"]
            org_coords = reverse_spec["org_coords"]
            vec_len = get_shape_list(org_input_mask)[0]
        else:
            max_num = tf.reduce_max(num_vec)  # []
            org_input_mask = generate_mask_based_on_lens(num_vec, max_num)
            org_coords = tf.where(org_input_mask)
            vec_len = get_shape_list(num_vec)[0]
        max_num =get_shape_list(org_input_mask)[-1]
        idx_mat = tf.tile(tf.expand_dims(tf.range(vec_len, dtype=tf.int32), -1), [1, max_num])
        num_indices = tf.expand_dims(tf.gather_nd(idx_mat, org_coords), -1) # [nbs,1]
        return tf.gather_nd(input_tensor, num_indices)  # [nbs,...]


def number_to_index(num_vec, name=None):  # [3, 2, 0, 2, 1] -> [0, 0, 0, 1, 1, 3, 3, 4]
    with tf.name_scope(name or "number_to_index"):
        vec_len = get_shape_list(num_vec)[0]
        max_num = tf.reduce_max(num_vec)  # []

        idx_mat = tf.tile(tf.expand_dims(tf.range(vec_len, dtype=tf.int32), -1), [1, max_num])  # [len,num]
        num_mask = generate_mask_based_on_lens(num_vec, max_num)  # [len,num]
        coords = tf.where(num_mask)  # [new,2]
        return tf.gather_nd(idx_mat, coords)  # [new]



if __name__ == '__main__':

    inp_tf = tf.constant([0, 0, 0, 0], dtype=tf.int32)
    out_tf = number_to_index(inp_tf)

    sess = tf.Session()

    print(sess.run(out_tf))




