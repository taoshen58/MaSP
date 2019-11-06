import tensorflow as tf
from functools import reduce
import math
import copy
import six
from operator import mul

VERY_BIG_NUMBER = 1e12
VERY_SMALL_NUMBER = 1e-12
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


# def assert_rank_lagacy(inp_tensor, rank_num):
#     assert len(inp_tensor.shape.as_list()) == rank_num, "tensor's rank should be \'%d\'" % rank_num

def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank.
        name: Optional name of the tensor for the error message.

    Raises:
        ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def add_pos_emb_idx(inp_token, n_vocab, name=None):
    with tf.name_scope(name or "add_pos_emb_idx"):
        sl = tf.shape(inp_token)[-1]
        sl_idxs = tf.range(n_vocab, n_vocab + sl)  # [sl]
        sl_idxs_tile = tf.ones_like(inp_token) * sl_idxs
        return tf.stack([inp_token, sl_idxs_tile], axis=-1)


def combine_sequences(sequences, axis=-1, name=None):
    with tf.name_scope(name or 'combine_sequences'):
        shapes = [shape_list(seq) for seq in sequences]
        sl_list = [shp[axis] for shp in shapes]
        sl_max = tf.reduce_max(tf.stack(sl_list))

        def _get_padding_shape(_shape, _sl, _sl_max, _dim):
            _new_shape = copy.copy(_shape)
            _new_shape[_dim] = _sl_max - _sl
            return _new_shape

        return tf.stack(
            [tf.concat(
                [
                    seq,
                    tf.zeros(_get_padding_shape(shp, sl, sl_max, axis), seq.dtype)
                ], axis=axis
            )
             for seq, shp, sl in zip(sequences, shapes, sl_list)],
            axis=axis
        )


def separate_sequences(inp_tensor, sl_list, axis, name=None):
    with tf.name_scope(name or 'separate_sequences'):
        assert shape_list(inp_tensor)[axis] == len(sl_list)
        split_tensors = [tf.squeeze(tsr, axis=[axis]) for tsr in tf.split(inp_tensor, len(sl_list), axis=axis)]
        sl_max = tf.shape(split_tensors[0])[axis]
        return tuple(
            tf.split(split_tensor, [sl, sl_max-sl], axis=axis)[0] for sl, split_tensor in zip(sl_list, split_tensors)
        )


def shape_list(x):  # read
    """
    deal with dynamic shape in tensorflow cleanly
    """
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def exp_mask_v3(val, m, multi_head=False, high_dim=False, name=None):
    """Multi-head dim on 2nd-dm"""
    with tf.name_scope(name or "new_exp_mask"):
        if multi_head:
            m = tf.expand_dims(m, 1)
        if high_dim:
            m = tf.expand_dims(m, -1)
        m_flt = tf.cast(m, tf.float32)
        return val + (1. - m_flt) * VERY_NEGATIVE_NUMBER


def mask_v3(val, m, multi_head=False, high_dim=False, name=None):
    with tf.name_scope(name or "new_exp_mask"):
        if multi_head:
            m = tf.expand_dims(m, 1)
        if high_dim:
            m = tf.expand_dims(m, -1)
        m_flt = tf.cast(m, val.dtype)
        return val * m_flt


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
        tensor: A tf.Tensor object to find the shape of.
        expected_rank: (optional) int. The expected rank of `tensor`. If this is
            specified and the `tensor` has a different rank, and exception will be
            thrown.
        name: Optional name of the tensor for the error message.

    Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
        be returned as python integers, and dynamic dimensions will be returned
        as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def get_last_state(rnn_out_put, mask): # correct
    '''
    get_last_state of rnn output
    :param rnn_out_put: [d1,d2,dn-1,max_len,d]
    :param mask: [d1,d2,dn-1,max_len]
    :return: [d1,d2,dn-1,d]
    '''
    rnn_out_put_flatten = flatten(rnn_out_put, 2)# [X, ml, d]
    mask_flatten = flatten(mask,1) # [X,ml]
    idxs = tf.reduce_sum(tf.cast(mask_flatten,tf.int32),-1) - 1 # [X]
    indices = tf.stack([tf.range(tf.shape(idxs)[0]), idxs], axis=-1) #[X] => [X,2]
    flatten_res = tf.expand_dims(tf.gather_nd(rnn_out_put_flatten, indices),-2 )# #[x,d]->[x,1,d]
    return tf.squeeze(reconstruct(flatten_res,rnn_out_put,2),-2) #[d1,d2,dn-1,1,d] ->[d1,d2,dn-1,d]


# ================= Dimension Operation =================
def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep, dim_reduced_keep=None):
    dim_reduced_keep = dim_reduced_keep or keep

    ref_shape = ref.get_shape().as_list() # original shape
    tensor_shape = tensor.get_shape().as_list() # current shape
    ref_stop = len(ref_shape) - keep # flatten dims list
    tensor_start = len(tensor_shape) - dim_reduced_keep  # start
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)] #
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))] #
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out


# =============== weight decay ===============
def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    for var in variables:
        if len(var.get_shape().as_list()) <= 1: continue
        tf.add_to_collection('reg_vars', var)
        counter += 1
    return counter


def add_var_reg(var):
    tf.add_to_collection('reg_vars', var)


def add_wd_for_var(var, wd):
    with tf.name_scope("weight_decay"):
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                   name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
        tf.add_to_collection('losses', weight_decay)


# ===== Activation Function =====
def gelu(x):  # read
    # return 0.5*x*(1+tf.tanh(math.sqrt(2/math.pi)*(x+0.044715*tf.pow(x, 3))))
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
    input_tensor: float Tensor to perform activation.

    Returns:
    `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0)))
    return x * cdf


def swish(x):  # read
    return x*tf.nn.sigmoid(x)


def selu(x):
    with tf.name_scope('elu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x >= 0.0, x, alpha*tf.nn.elu(x))


def activation_name_to_func(activation_name):
    assert isinstance(activation_name, str)
    if isinstance(activation_name, str):
        if activation_name == 'linear':
            act_fn = tf.identity
        elif activation_name == 'relu':
            act_fn = tf.nn.relu
        elif activation_name == 'elu':
            act_fn = tf.nn.elu
        elif activation_name == 'selu':
            act_fn = selu
        elif activation_name == 'sigmoid':
            act_fn = tf.nn.sigmoid
        elif activation_name == 'tanh':
            act_fn = tf.nn.tanh
        elif activation_name == 'exp':
            act_fn = tf.exp
        elif activation_name == 'log':
            act_fn = tf.log
        elif activation_name == 'gelu':
            act_fn = gelu
        elif activation_name == 'swish':
            act_fn = swish
        elif activation_name == 'lrelu':
            act_fn = tf.nn.leaky_relu
        else:
            raise AttributeError('no activation function named as %s' % activation_name)
    elif hasattr(activation_name, '__call__'):  # callable
        act_fn = activation_name
    else:
        raise AttributeError
    return act_fn

def act_name2fn(afn):
    return activation_name_to_func(afn)
