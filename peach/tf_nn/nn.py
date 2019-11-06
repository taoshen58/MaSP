"""
Note:
    1. All multi-head fn: the head dim always on the 2nd
    2.
"""
import tensorflow as tf
from peach.tf_nn.general import assert_rank, shape_list, act_name2fn, flatten, reconstruct, add_reg_without_bias,\
    exp_mask_v3, mask_v3, get_shape_list
from functools import reduce
from operator import mul


def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        if is_train is None:
            if keep_prob < 1.0:
                return tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
        else:
            if keep_prob < 1.0:
                out = tf.cond(
                    is_train,
                    lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed),
                    lambda: x
                )
                return out
        return x


def get_emb_matrix(n_vocab, n_ctx, embd_dim, is_train, embd_dropout, scope=None):
    with tf.variable_scope(scope):
        pad_emb = tf.zeros([1, embd_dim], tf.float32, 'pad_emb')
        we = tf.get_variable("we", [n_vocab-1+n_ctx, embd_dim])
        we_dp = dropout(we, embd_dropout, is_train)
        we_vocab = we_dp[:n_vocab-1]
        return tf.concat([pad_emb, we_dp], axis=0), we_vocab


# ===================  multi-head ====================
def split_head(inp_tensor, head_num, name=None):
    with tf.name_scope(name or 'split_head'):
        # [bs,sl,num] as an example
        inp_shape = get_shape_list(inp_tensor)  # [3] for [bs,sl,hn]
        # head params
        hn = inp_shape[-1]
        assert hn % head_num == 0
        head_dim = hn // head_num
        new_input_shape = inp_shape[:-1] + [head_num, head_dim] # [4] for [bs,sl,hd_num,hd_dim]

        new_perm = list(range(len(new_input_shape)))  # [0,1,2,3]
        head_dim = new_perm.pop(-2) # [0,1,3]
        new_perm.insert(1, head_dim)  # [0,2,1,3]

        inp_tensor_hd = tf.reshape(inp_tensor, new_input_shape)  # [bs,sl,hd_num,hd_dim]
        return tf.transpose(inp_tensor_hd, new_perm)  # [bs,hd_num,sl,hd_dim]


def combine_head(inp_tensor, name=None):
    with tf.name_scope(name or 'combine_head'):
        # [bs,hd_num,sl,hd_dim] as an example
        inp_shape = get_shape_list(inp_tensor)  # [4] for [bs,hd_num,sl,hd_dim]

        # get hn from head_num * head_dim
        assert isinstance(inp_shape[1], int) and isinstance(inp_shape[-1], int)
        hn = inp_shape[1] * inp_shape[-1]

        # move head dim to -1
        new_perm = list(range(len(inp_shape)))  # [0,1,2,3]
        head_dim = new_perm.pop(1)  # [0,2,3]
        new_perm.insert(-1, head_dim)  # [0,2,1,3]

        inp_tensor_new_perm = tf.transpose(inp_tensor, new_perm)  # [bs,sl,hd_num,hd_dim]
        # get new shape
        new_shape = get_shape_list(inp_tensor_new_perm)[:-2] + [hn]  # [3] for [bs,sl,hn]
        # return reshaped tensor
        return tf.reshape(inp_tensor_new_perm, new_shape)  # [bs,sl,hn]


def bn_dense_layer_multi_head(
        input_tensor, hn, bias, bias_start=0.0, scope=None, activation='relu',
        enable_bn=False, wd=0., keep_prob=1.0, is_train=None, dup_num=1, merge_var=False
):
    assert not enable_bn
    """The input could be >3-d and the 1d-for bs, 2d for head, -1d for hn"""

    act_fn = act_name2fn(activation)

    with tf.variable_scope(scope or 'bn_dense_layer_multi_head'):
        input_tensor = dropout(input_tensor, keep_prob, is_train)  # dropout [bs,hd,sl,dim]
        # the comments using 4d [bs,hd,sl,dim] for example
        input_shape = get_shape_list(input_tensor)  # [4] for [bs,hd,sl,dim]
        assert len(input_shape) >= 3
        # exchange 1st and 2nd dimension
        perm_t = list(range(len(input_shape)))  # [0,1,2,3]
        perm_t[0], perm_t[1] = perm_t[1], perm_t[0]  # [1,0,2,3]
        input_tensor_t = tf.transpose(input_tensor, perm_t)  # [hd,bs,sl,dim]

        # merge and reshape
        input_shape_t = get_shape_list(input_tensor_t)  # [4] for [hd,bs,sl,dim]
        dims_merge = input_shape_t[1:-1]  # [2] for [bs,sl]
        new_dim = reduce(mul, dims_merge)  # bs*sl
        new_shape = [input_shape_t[0], new_dim, input_shape_t[-1]]  # [3] for [hd,bs*sl,dim]
        input_tensor_rsp = tf.reshape(input_tensor_t, new_shape)  # [hd,bs*sl,dim]

        # dense layer
        hd_num = new_shape[0]  # head num
        hd_dim = new_shape[-1]  # head dim

        if merge_var:
            weight = tf.get_variable('W', shape=[hd_num, hd_dim, hn*dup_num])
        else:
            weight_list = []
            for i in range(hd_num):
                sub_weight_list = []
                for j in range(dup_num):
                    sub_weight_list.append(tf.get_variable('W_%d_%d' % (i, j), shape=[hd_dim, hn]))
                weight_list.append(tf.concat(sub_weight_list, -1) if dup_num > 1 else sub_weight_list[0])
            weight = tf.stack(weight_list, 0)

        out_rsp = tf.matmul(input_tensor_rsp, weight)  # hd_num, bs*sl, hn
        if bias:
            if merge_var:
                bias_val = tf.get_variable(
                        'bias', shape=[hd_num, 1, hn], dtype=tf.float32,
                        initializer=tf.constant_initializer(bias_start))
            else:
                bias_list = []
                for i in range(hd_num):
                    sub_bias_list = []
                    for j in range(dup_num):
                        sub_bias_list.append(
                            tf.get_variable(
                                'bias_%d_%d' % (i, j), shape=[1, hn], dtype=tf.float32,
                                initializer=tf.constant_initializer(bias_start)))
                    bias_list.append(tf.concat(sub_bias_list, -1) if dup_num > 1 else sub_bias_list[0])
                bias_val = tf.stack(bias_list, 0)
            out_rsp = out_rsp + bias_val   # hd_num, bs*sl, hn

        # un-merge
        output_shape_t = [new_shape[0]] + dims_merge + [hn]  # [4] for [hd,bs,sl,new_dim]
        output_t = tf.reshape(out_rsp, output_shape_t)  # [hd,bs,sl,new_dim]

        # transpose
        output = tf.transpose(output_t, perm_t)  # [bs,hd,sl,new_dim]

        if wd:
            tf.add_to_collection('reg_vars', weight)

        return act_fn(output)


# ============== sequence compression ===============
def pooling_with_mask(rep_tensor, rep_mask, method='max', scope=None):
    # rep_tensor have one more rank than rep_mask
    with tf.name_scope(scope or '%s_pooling' % method):
        if method == 'max':
            rep_tensor_masked = exp_mask_v3(rep_tensor, rep_mask, high_dim=True)
            output = tf.reduce_max(rep_tensor_masked, -2)
        elif method == 'mean':
            rep_tensor_masked = mask_v3(rep_tensor, rep_mask, high_dim=True)  # [...,sl,hn]
            rep_sum = tf.reduce_sum(rep_tensor_masked, -2)  # [..., hn]
            denominator = tf.reduce_sum(tf.cast(rep_mask, tf.int32), -1, True)  # [..., 1]
            denominator = tf.where(tf.equal(denominator, tf.zeros_like(denominator, tf.int32)),
                                   tf.ones_like(denominator, tf.int32),
                                   denominator)
            output = rep_sum / tf.cast(denominator, tf.float32)
        else:
            raise AttributeError('No Pooling method name as %s' % method)
        return output


# todo: source2token self-attention network


# =============== Deeper Network ===============
def residual_connection_with_dense(
        x, hn, bias, bias_start=0.0, scope=None,
        activation='relu', enable_bn=False,
        wd=0., keep_prob=1.0, is_train=None,
        residual_keep_prob=1.):
    with tf.variable_scope(scope or 'residual_connection_with_dense'):
        y1 = bn_dense_layer_v2(
            x, hn, bias, bias_start, "dense_layer_1", activation, enable_bn, wd, keep_prob, is_train
        )
        y2 = bn_dense_layer_v2(
            y1, get_shape_list(x)[-1], bias, bias_start, "dense_layer_2", "linear", enable_bn, wd, keep_prob, is_train
        )

        return residual_connection(x, y2, is_train, residual_keep_prob, 'residual_connection')



def residual_connection(x, y, is_train=None, residual_keep_prob=1., scope=None):
    with tf.variable_scope(scope or 'residual_connection'):
        y = dropout(y, residual_keep_prob, is_train)
        return layer_norm(x + y, scope='layer_norm')


def layer_norm(inputs, epsilon=1e-6, scope=None):
    with tf.variable_scope(scope or "layer_norm"):
        channel_size = inputs.get_shape().as_list()[-1]

        scale = tf.get_variable("scale", shape=[channel_size],
                                initializer=tf.ones_initializer())

        offset = tf.get_variable("offset", shape=[channel_size],
                                 initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(inputs, axis=-1, keep_dims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1,
                                  keep_dims=True)

        norm_inputs = (inputs - mean) * tf.rsqrt(variance + epsilon)

        return norm_inputs * scale + offset


def highway_layer(
        input_tensor, bias, bias_start=0.0, scope=None, activation='relu', enable_bn=False,
        wd=0., keep_prob=1.0, is_train=None):
    ivec = input_tensor.get_shape().as_list()[-1]
    with tf.variable_scope(scope or "highway_layer"):
        trans = bn_dense_layer_v2(
            input_tensor, ivec, bias, bias_start, 'map', activation, enable_bn, wd, keep_prob, is_train)
        gate = bn_dense_layer_v2(
            input_tensor, ivec, bias, bias_start, 'gate', 'linear', enable_bn, wd, keep_prob, is_train)
        gate = tf.nn.sigmoid(gate)
        out = gate * trans + (1 - gate) * input_tensor
        return out


def highway_network(
        num_layers, input_tensor, hn=None,
        bias=True, bias_start=0.0,
        scope=None, activation='relu', enable_bn=False,
        wd=0., keep_prob=1.0, is_train=None
):
    with tf.variable_scope(scope or highway_network):
        if isinstance(hn, int):
            input_tensor = bn_dense_layer_v2(
                input_tensor, hn, bias, bias_start, 'mapping', activation, enable_bn, wd, keep_prob, is_train
            )

        for i in range(num_layers):
            input_tensor = highway_layer(
                input_tensor, bias, bias_start, "{}_layer".format(i), activation, enable_bn, wd, keep_prob, is_train
            )
        return input_tensor


# =================== linear layer ===================
def bn_dense_layer_v2(
        input_tensor, hn, bias, bias_start=0.0, scope=None,
        activation='relu', enable_bn=False,
        wd=0., keep_prob=1.0, is_train=None, dup_num=1, merge_var=False
):
    act_fn = act_name2fn(activation)
    with tf.variable_scope(scope or 'bn_dense_layer'):
        input_tensor = dropout(input_tensor, keep_prob, is_train)
        # the comment use a 3d tensor [bs,sl,hn] as a example
        input_shape = get_shape_list(input_tensor)  # [3]
        assert len(input_shape) >= 2  # at least [bs,hn]
        # merge
        dims_merge = input_shape[:-1]  # [all unrelated dims]
        new_dim = reduce(mul, dims_merge)  # get the merged dim
        new_shape = [new_dim, input_shape[-1]]  # new shape for matmul [2]
        input_tensor_rsp = tf.reshape(input_tensor, new_shape)  #  [xx,dim]

        # dense layer
        input_dim = new_shape[-1]
        if merge_var:
            weight = tf.get_variable('W', shape=[input_dim, hn * dup_num], dtype=tf.float32)
        else:
            weight_list = []
            for i in range(dup_num):
                weight_list.append(tf.get_variable('W_%d' % i, shape=[input_dim, hn]))
            weight = tf.concat(weight_list, -1)
        output_rsp = tf.matmul(input_tensor_rsp, weight)

        if bias:
            if merge_var or dup_num == 1:
                bias_val = tf.get_variable(
                    'bias', shape=[hn * dup_num], dtype=tf.float32,
                    initializer=tf.constant_initializer(bias_start)
                )
            else:
                bias_list = []
                for i in range(dup_num):
                    bias_list.append(
                        tf.get_variable(
                            'bias_%d' % i, shape=[hn], dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_start))
                    )
                bias_val = tf.concat(bias_list, -1)
            output_rsp += bias_val

        # output reshape
        output_shape = dims_merge + [hn * dup_num]  # [3] for [bs,sl,new_hn]
        output = tf.reshape(output_rsp, output_shape)  # [bs,sl,new_hn]

        if enable_bn:
            output = tf.contrib.layers.batch_norm(
                output, center=True, scale=True, is_training=is_train,
                updates_collections=None,  decay=0.9,
                scope='bn')

        if wd:
            tf.add_to_collection('reg_vars', weight)

        return act_fn(output)


def conv1d(
        x, scope, nf, rf, w_init=tf.random_normal_initializer(stddev=0.02),
        b_init=tf.constant_initializer(0), pad='VALID', train=None):
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        w = tf.get_variable("w", [rf, nx, nf], initializer=w_init)
        b = tf.get_variable("b", [nf], initializer=b_init)
        if rf == 1:  # faster 1x1 conv: use 'matmul' or 'conv1d'
            c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, shape_list(x)[:-1]+[nf])
        else:  # was used to train LM
            c = tf.nn.conv1d(x, w, stride=1, padding=pad)+b
        return c


def bn_dense_layer(input_tensors, hn, bias, bias_start=0.0, scope=None,
                   activation='relu', enable_bn=False,
                   wd=0., keep_prob=1.0, is_train=None):
    tf.logging.warning("Please use \"bn_dense_layer_v2\" rather than \"bn_dense_layer\" for future support! ")
    with tf.variable_scope(scope or 'bn_dense_layer'):
        linear_map = linear(input_tensors, hn, bias, bias_start, 'linear_map',
                            False, wd, keep_prob, is_train)
        if enable_bn:
            linear_map = tf.contrib.layers.batch_norm(
                linear_map, center=True, scale=True, is_training=is_train,
                updates_collections=None, decay=0.9,
                scope='bn')
        act_fn = act_name2fn(activation)
        return act_fn(linear_map)


def _linear(xs, output_size, bias, bias_start=0., scope=None):
    with tf.variable_scope(scope or 'linear_layer'):
        x = tf.concat(xs, -1)
        input_size = x.get_shape()[-1]
        W = tf.get_variable('W', shape=[input_size, output_size], dtype=tf.float32)
        if bias:
            bias = tf.get_variable('bias', shape=[output_size], dtype=tf.float32,
                                   initializer=tf.constant_initializer(bias_start))
            out = tf.matmul(x, W) + bias
        else:
            out = tf.matmul(x, W)
        return out


def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None):
    if args is None or (isinstance(args, (tuple, list)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (tuple, list)):
        args = [args]

    flat_args = [flatten(arg, 1) for arg in args]  # for dense layer [(-1, d)]
    if input_keep_prob < 1.0:
        assert is_train is not None
        # for dense layer [(-1, d)]
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)
                     for arg in flat_args]
    flat_out = _linear(flat_args, output_size, bias, bias_start=bias_start, scope=scope)  # dense
    out = reconstruct(flat_out, args[0], 1)  # ()
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])

    if wd:
        add_reg_without_bias()

    return out


# ================== BILINEAR ================
# def bilinear_classifier_nary(input1, input2, n_classes, keep_prob, add_bias1=True, add_bias2=True):
#     input_shape1 = get_shape_list(input1)
#     input_shape2 = get_shape_list(input2)

def bilinear_classifier_nary(
        inputs1, inputs2, n_classes, keep_prob=1., is_training=None, add_bias1=True, add_bias2=True):
    """"""

    input_shape1 = tf.shape(inputs1)
    input_shape2 = tf.shape(inputs2)

    batch_size1 = input_shape1[0]
    batch_size2 = input_shape2[0]

    # with tf.control_dependencies([tf.assert_equal(input_shape1[1], input_shape2[1])]):
    bucket_size1 = input_shape1[1]
    bucket_size2 = input_shape2[1]
    input_size1 = inputs1.get_shape().as_list()[-1]
    input_size2 = inputs2.get_shape().as_list()[-1]

    input_shape_to_set1 = [tf.Dimension(None), tf.Dimension(None), input_size1 + 1]
    input_shape_to_set2 = [tf.Dimension(None), tf.Dimension(None), input_size2 + 1]

    if isinstance(keep_prob, tf.Tensor) or keep_prob < 1:
        noise_shape1 = tf.stack([batch_size1, 1, input_size1])
        noise_shape2 = tf.stack([batch_size2, 1, input_size2])

        inputs1 = dropout(inputs1, keep_prob, is_training, noise_shape=noise_shape1)
        inputs2 = dropout(inputs2, keep_prob, is_training, noise_shape=noise_shape2)

    inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size1, bucket_size1, 1]))])
    inputs1.set_shape(input_shape_to_set1)
    inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size2, bucket_size2, 1]))])
    inputs2.set_shape(input_shape_to_set2)

    bilin = bilinear(inputs1, inputs2,
                   n_classes,
                   add_bias1=add_bias1,
                   add_bias2=add_bias2,
                   initializer=tf.zeros_initializer())

    return bilin


def bilinear(inputs1, inputs2, output_size, add_bias2=True, add_bias1=True, add_bias=False, initializer=None):
    """"""

    with tf.variable_scope('Bilinear'):
        # Reformat the inputs
        ndims = len(inputs1.get_shape().as_list())
        inputs1_shape = tf.shape(inputs1)
        inputs1_bucket_size = inputs1_shape[ndims - 2]
        inputs1_size = inputs1.get_shape().as_list()[-1]

        inputs2_shape = tf.shape(inputs2)
        inputs2_bucket_size = inputs2_shape[ndims - 2]
        inputs2_size = inputs2.get_shape().as_list()[-1]
        # output_shape = []
        batch_size1 = 1
        batch_size2 = 1
        for i in range(ndims - 2):
            batch_size1 *= inputs1_shape[i]
            batch_size2 *= inputs2_shape[i]
        # output_shape.append(inputs1_shape[i])
        # output_shape.append(inputs1_bucket_size)
        # output_shape.append(output_size)
        # output_shape.append(inputs2_bucket_size)
        # output_shape = tf.stack(output_shape)
        inputs1 = tf.reshape(inputs1, tf.stack([batch_size1, inputs1_bucket_size, inputs1_size]))
        inputs2 = tf.reshape(inputs2, tf.stack([batch_size2, inputs2_bucket_size, inputs2_size]))
        if add_bias1:
            inputs1 = tf.concat(axis=2, values=[inputs1, tf.ones(tf.stack([batch_size1, inputs1_bucket_size, 1]))])
        if add_bias2:
            inputs2 = tf.concat(axis=2, values=[inputs2, tf.ones(tf.stack([batch_size2, inputs2_bucket_size, 1]))])

        # Get the matrix
        if initializer is None:
            # mat = orthonormal_initializer(inputs1_size + add_bias1, inputs2_size + add_bias2)[:, None, :]
            # mat = np.concatenate([mat] * output_size, axis=1)
            # initializer = tf.constant_initializer(mat)
            initializer = tf.initializers.orthogonal
        weights = tf.get_variable('Weights', [inputs1_size + add_bias1, output_size, inputs2_size + add_bias2],
                                  initializer=initializer)
        # tf.add_to_collection('Weights', weights)

        # inputs1: num_triggers_in_batch x 1 x self.trigger_mlp_size
        # inputs2: batch x seq_len x self.role_mlp_size

        # Do the multiplications
        # (bn x d) (d x rd) -> (bn x rd)
        lin = tf.matmul(tf.reshape(inputs1, [-1, inputs1_size + add_bias1]), tf.reshape(weights, [inputs1_size + add_bias1, -1]))
        # (b x nr x d) (b x n x d)T -> (b x nr x n)
        lin_reshape = tf.reshape(lin, tf.stack([batch_size1, inputs1_bucket_size * output_size, inputs2_size + add_bias2]))
        bilin = tf.matmul(lin_reshape, inputs2, adjoint_b=True)
        # (bn x r x n)
        bilin = tf.reshape(bilin, tf.stack([-1, output_size, inputs2_bucket_size]))

        # Get the bias
        if add_bias:
            bias = tf.get_variable('Biases', [output_size], initializer=tf.zeros_initializer())
            bilin += tf.expand_dims(bias, 1)

        return bilin


# ================ LOSS FUNCTION ============
def smoothed_softmax_cross_entropy_with_logits(**kwargs):
    logits = kwargs.get("logits")
    labels = kwargs.get("labels")
    label_smoothing = kwargs.get("label_smoothing") or 0.0
    normalize = kwargs.get("normalize")

    if logits is None or labels is None:
        raise ValueError("Both logits and labels must be provided")

    if not label_smoothing:
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels
        )
        return ce

    # adaptive for any rank
    vocab_size = get_shape_list(logits)[-1]

    n = tf.to_float(vocab_size - 1)
    p = 1.0 - label_smoothing
    q = label_smoothing / n

    soft_targets = tf.one_hot(tf.cast(labels, tf.int32), depth=vocab_size,
                              on_value=p, off_value=q)

    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=soft_targets)

    if not normalize:
        return xentropy

    normalizing = -(p * tf.log(p) + n * q * tf.log(q + 1e-20))

    return xentropy - normalizing


# ========= higher level network ===========
def language_model(org_inp_emb, org_input_token, org_inp_mask, tied_we_vocab):
    assert_rank(org_inp_emb, 3)
    with tf.name_scope('language_model_logits'):
        embd_dim = org_inp_emb.shape.as_list()[-1]
        sl = tf.shape(org_input_token)[-1]

        inp_emb = org_inp_emb[..., :-1, :]  # [bs,sl-1,dim]
        tgt_token = org_input_token[..., 1:]  # [bs,sl-1]
        tgt_token_mask = org_inp_mask[..., 1:]  # [bs,sl-1]

        #
        inp_emb_rsp = tf.reshape(inp_emb, [-1, embd_dim])  # [bs*(sl-1),dim]
        tgt_token_rsp = tf.reshape(tgt_token, [-1])  # [bs*(sl-1)]
        tgt_token_mask_rsp = tf.reshape(tgt_token_mask, [-1])  # [bs*(sl-1)]

        tgt_token_mask_rsp2_bl = tf.reshape(tgt_token_mask, [-1, sl-1])
        tgt_token_mask_rsp2_ft = tf.cast(tgt_token_mask_rsp2_bl, tf.float32)  # [bs,sl-1]

        lm_logits = tf.matmul(inp_emb_rsp, tied_we_vocab, transpose_b=True)  # [bs*(sl-1),voc-1]
        lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(  # bs*(sl-1)
            logits=lm_logits,
            labels=(tgt_token_rsp - 1) * tf.cast(tgt_token_mask_rsp, tf.int32),  # eliminate the padding index
        )
        lm_losses = tf.reshape(lm_losses, [-1, sl-1])  # bs,sl-1
        # [sent]
        lm_losses = tf.reduce_sum(lm_losses * tgt_token_mask_rsp2_ft, -1) / tf.reduce_sum(tgt_token_mask_rsp2_ft, -1)
        return lm_losses


# ================= MASKED SPARSE =================
def masked_dense2sparse(input_tensor, input_mask, name=None):
    with tf.variable_scope(name or "masked_dense2sparse"):
        coords = tf.where(input_mask)  # [n,ndim]
        sparse_res = tf.gather_nd(  # n,hn
            input_tensor, coords,
        )

        reverse_spec = {
            "org_input_mask": input_mask,
            "org_coords": coords
        }
        return sparse_res, reverse_spec


def masked_sparse2dense(input_tensor, reverse_spec, name=None):
    org_input_mask = reverse_spec['org_input_mask']
    org_coords = reverse_spec['org_coords']

    with tf.variable_scope(name or "masked_sparse2dense"):
        hn = get_shape_list(input_tensor)[-1]
        org_shape = get_shape_list(org_input_mask)
        org_shape.append(hn)
        return tf.scatter_nd(org_coords, input_tensor, org_shape)  # [xx,hn]


if __name__ == '__main__':
    vals = tf.constant(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ], tf.int32
    )

    input_tensor = tf.tile(tf.expand_dims(tf.to_float(vals), -1), [1, 1, 3]) * 2
    input_mask = tf.cast(vals, tf.bool)

    sparse_res, reverse_spec = masked_dense2sparse(input_tensor, input_mask)

    dense_res = masked_sparse2dense(sparse_res, reverse_spec)

    sess = tf.Session()

    sparse_res_np, dense_res_np = sess.run([sparse_res, dense_res])
    dense_res_np = dense_res_np[..., 0]
    print(sparse_res_np)
    print(dense_res_np)

