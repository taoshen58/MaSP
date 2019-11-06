import tensorflow as tf
import numpy as np
from peach.tf_nn.general import assert_rank, shape_list, act_name2fn, activation_name_to_func
from peach.tf_nn.nn import dropout, conv1d
from peach.tf_nn.training import get_trainable_vars
from peach.tf_nn.attn import mask_ft_generation
import json
import os
import math


# ================= OpenAI-style OpenAI transformer ============
def multi_head_block(
        x, scope, train=None, scale=False, n_head=12, afn='gelu',
        resid_dropout=0.9, attn_dropout=0.9, reuse=None,
        use_global=False, use_direction=False, x_mask=None, global_afn='exp', attn_self=True,
):
    assert_rank(x, 3)
    with tf.variable_scope(scope, reuse=reuse):
        # b gene
        if use_direction:
            b = tf.transpose(mask_ft_generation(x_mask, n_head, True, attn_self=attn_self), [1, 0, 2, 3])
        else:
            b = None

        nx = shape_list(x)[-1]

        a = multi_head_w_global(
            x, 'attn', nx, n_head, train=train, scale=scale,
            resid_dropout=resid_dropout, attn_dropout=attn_dropout,
            use_global=use_global, use_direction=use_direction, b=b, global_afn=global_afn
        )
        n = norm(x+a, 'ln_1_openai_trans')
        m = mlp(n, 'mlp', nx*4, train=train, afn=afn, resid_dropout=resid_dropout)
        h = norm(n+m, 'ln_2_openai_trans')
        return h


def multi_head_w_global(  # Added and Modified by xxx xxx
        x, scope, n_state, n_head, train=None, scale=False, resid_dropout=0.9, attn_dropout=0.9,
        use_global=False, use_direction=False, b=None, global_afn='exp',
):
    assert n_state % n_head == 0
    with tf.variable_scope(scope):
        sl = shape_list(x)[-2]
        if not use_direction:
            b = tf.matrix_band_part(tf.ones([sl, sl]), -1, 0)  # Lower triangular part.
            b = tf.reshape(b, [1, 1, sl, sl])

        c = conv1d(x, 'c_attn_openai_trans', n_state * 3, 1, train=train)
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)  # bs,hd,sl,d
        k = split_heads(k, n_head, k=True)  # bs,hd,d,sl
        v = split_heads(v, n_head)  # bs,hd,sl,d

        # 1. t2t
        w = tf.matmul(q, k)  # bs,hd,sl, sl
        if scale:
            n_state_hd = shape_list(v)[-1]
            w = w * tf.rsqrt(tf.cast(n_state_hd, tf.float32))

        if use_global:
            e_w = activation_name_to_func(global_afn)(w) * b

            # 2. s2t
            w_g = split_heads(conv1d(x, "c_w_g", n_state, 1, train=train), n_head)  # bs,hd,sl,d
            e_w_g = tf.exp(w_g)  # # bs,hd,sl,d

            # 3. mtsa
            accum_z_deno = tf.matmul(e_w, e_w_g)  # bs,hd,sl,dim
            accum_z_deno = tf.where(  # in case of NaN and Inf
                tf.greater(accum_z_deno, tf.zeros_like(accum_z_deno)),
                accum_z_deno,
                tf.ones_like(accum_z_deno)
            )
            e_w = dropout(e_w, math.sqrt(attn_dropout), train)
            e_w_g = dropout(e_w_g, math.sqrt(attn_dropout), train)
            rep_mul_score = v * e_w_g
            accum_rep_mul_score = tf.matmul(e_w, rep_mul_score)
            a = accum_rep_mul_score / accum_z_deno
        else:
            w = w * b + -1e9 * (1 - b)
            w = tf.nn.softmax(w)
            w = w * b  # fixed the bug
            w = dropout(w, attn_dropout, train)  # attention dropout
            a = tf.matmul(w, v)

        a = merge_heads(a)
        a = conv1d(a, 'c_proj_openai_trans', n_state, 1, train=train)
        a = dropout(a, resid_dropout, train, )
        return a


def multi_head(x, scope, n_state, n_head, train=None, scale=False, resid_dropout=0.9, attn_dropout=0.9):
    assert n_state % n_head == 0
    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn_openai_trans', n_state * 3, 1, train=train)  # position-wise fully-connected layer
        q, k, v = tf.split(c, 3, 2)
        q = split_heads(q, n_head)
        k = split_heads(k, n_head, k=True)
        v = split_heads(v, n_head)
        a = _attn(q, k, v, train=train, scale=scale, attn_dropout=attn_dropout)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj_openai_trans', n_state, 1, train=train)
        a = dropout(a, resid_dropout, train, )
        return a


def _attn(q, k, v, train=None, scale=False, attn_dropout=0.9):  # read
    w = tf.matmul(q, k)

    if scale:
        n_state = shape_list(v)[-1]
        w = w*tf.rsqrt(tf.cast(n_state, tf.float32))

    w = mask_attn_weights(w)  # highlight, this is uni-directional self-attention
    w = tf.nn.softmax(w)

    # w = tf.Print(w, [tf.shape(w)])

    w = dropout(w, attn_dropout, train)  # attention dropout

    a = tf.matmul(w, v)
    return a


def mlp(x, scope, n_state, train=None, afn='gelu', resid_dropout=0.9):  # read: 3layer mlp
    with tf.variable_scope(scope):
        nx = shape_list(x)[-1]
        act = act_name2fn(afn)
        h = act(conv1d(x, 'c_fc_openai_trans', n_state, 1, train=train))
        h2 = conv1d(h, 'c_proj_openai_trans', nx, 1, train=train)
        h2 = dropout(h2, resid_dropout, train)
        return h2


def mask_attn_weights(w):  # read
    n = shape_list(w)[-1]
    b = tf.matrix_band_part(tf.ones([n, n]), -1, 0)  # Lower triangular part.
    b = tf.reshape(b, [1, 1, n, n])
    w = w*b + -1e9*(1-b)
    return w


def split_heads(x, n, k=False):  # read
    if k:
        return tf.transpose(split_states(x, n), [0, 2, 3, 1])  # plus transpose
    else:
        return tf.transpose(split_states(x, n), [0, 2, 1, 3])


def split_states(x, n):  # read
    x_shape = shape_list(x)
    m = x_shape[-1]
    new_x_shape = x_shape[:-1]+[n, m//n]
    return tf.reshape(x, new_x_shape)


def merge_heads(x):  # read
    return merge_states(tf.transpose(x, [0, 2, 1, 3]))


def merge_states(x):  # read
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
    return tf.reshape(x, new_x_shape)


def _norm(x, g=None, b=None, e=1e-5, axis=[1]):  # read
    u = tf.reduce_mean(x, axis=axis, keep_dims=True)
    s = tf.reduce_mean(tf.square(x-u), axis=axis, keep_dims=True)
    x = (x - u) * tf.rsqrt(s + e)
    if g is not None and b is not None:
        x = x*g + b
    return x


def norm(x, scope, axis=[-1]):  # read
    with tf.variable_scope(scope):
        n_state = shape_list(x)[-1]
        g = tf.get_variable("g", [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state], initializer=tf.constant_initializer(0))
        g, b = get_ema_vars(g, b)
        return _norm(x, g, b, axis=axis)


def get_ema_vars(*vs):
    if tf.get_variable_scope().reuse:
        gvs = tf.global_variables()
        vs = [get_ema_if_exists(v, gvs) for v in vs]
    if len(vs) == 1:
        return vs[0]
    else:
        return vs


def get_ema_if_exists(v, gvs):
    name = v.name.split(':')[0]
    ema_name = name+'/ExponentialMovingAverage:0'
    ema_v = [v for v in gvs if v.name == ema_name]
    if len(ema_v) == 0:
        ema_v = [v]
    return ema_v[0]


# ========== For OpenAI transformer ====================
def get_transformer_clf_features(inp_emb, inp_token, clf_token):
    assert_rank(inp_emb, 3)
    with tf.name_scope('get_transformer_clf_features'):
        bs, sl, embd_dim = shape_list(inp_emb)

        bs_idxs = tf.range(0, bs)  # [bs]
        sent_idxs = tf.argmax(tf.cast(tf.equal(inp_token, clf_token), tf.float32), -1)  # bs
        feature_idxs = tf.stack([bs_idxs, tf.cast(sent_idxs, tf.int32)], -1)  # [bs,2]
        return tf.gather_nd(inp_emb, feature_idxs)  # [bs,inp_dim]


def load_openai_pretrained_transformer_model(
        n_ctx, n_special, embd_dim, n_layer,
        sess, pretrain_dir,
        current_scope,
        openai_vars_filters=('openai_trans',)
):
    openai_vars_filters = openai_vars_filters \
        if not isinstance(openai_vars_filters, str) else (openai_vars_filters,)
    openai_vars = get_trainable_vars(current_scope, keys=openai_vars_filters)

    shapes = json.load(open(os.path.join(pretrain_dir, 'params_shapes.json')))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(os.path.join(pretrain_dir, 'params_{}.npy'.format(n))) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    init_params[0] = init_params[0][:n_ctx]
    init_params[0] = np.concatenate(
        [
            init_params[1],
            (np.random.randn(n_special, embd_dim) * 0.02).astype(np.float32),
            init_params[0]],
        0)
    del init_params[1]

    if n_layer == -1:
        n_transfer = 0
    else:
        n_transfer = 1 + n_layer * 12

    sess.run([p.assign(ip) for p, ip in zip(openai_vars[:n_transfer], init_params[:n_transfer])])


# ================== Optimizer ================
def openai_adam(
        params, grads, lr, schedule, t_total, b1=0.9, b2=0.999, e=1e-8, l2=0,
        vector_l2=False, max_grad_norm=-1, **kwargs):
    """
    adam with weight decay fix
    """
    t = tf.Variable(0, dtype=tf.float32, trainable=False)
    tt = t+1
    updates = [t.assign(tt)]

    if "global_step" in kwargs:
        ngs = kwargs['global_step'] + 1
        updates.append(kwargs['global_step'].assign(ngs))

    if max_grad_norm > 0:
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    for p, g in zip(params, grads):
        if p is None or g is None:
            print("can't train", p.name, g)
        else:
            if isinstance(g, tf.IndexedSlices):
                g = tf.convert_to_tensor(g)
            m = tf.Variable(p*0, dtype=tf.float32, trainable=False)
            v = tf.Variable(p*0, dtype=tf.float32, trainable=False)
            lrt = lr*tf.sqrt(1-b2**tt)/(1-b1**tt)
            lrt *= schedule(t/t_total)
            mt = b1*m + (1-b1)*g
            vt = b2*v + (1-b2)*g*g
            if (len(p.get_shape()) > 1 or vector_l2) and l2 > 0:
                pt = p - lrt * (mt / (tf.sqrt(vt) + e) + l2*p)
            else:
                pt = p - lrt * (mt / (tf.sqrt(vt) + e))
            updates.extend([m.assign(mt), v.assign(vt), p.assign(pt)])
    return tf.group(*updates)


def warmup_cosine(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return s*(x/warmup) + (1-s)*(0.5 * (1 + tf.cos(math.pi * x)))

def warmup_constant(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return s*(x/warmup) + (1-s)*1

def warmup_linear(x, warmup=0.002):
    s = tf.cast(x <= warmup, tf.float32)
    return (s*(x/warmup) + (1-s))*(1-x)

def warmup_exp(x, warmup=0.002, min_val=0.01):
    s = tf.cast(x <= warmup, tf.float32)
    decay = min_val ** (1./100)
    num = tf.floor(x * 100)
    return s*(x/warmup) + (1-s) * 1. * (decay ** num)

def warmup_exp5(x, warmup=0.002):
    return warmup_exp(x, warmup, 0.05)


def warmup_new(x, k, m, n, warmup=0.002):
    # k, m, n = 9, 6, 3
    a0 = 1.0 * k / (k + m - n)
    a1 = (10.-k) / 10
    a2 = a1 * n / m

    s = tf.cast(x <= warmup, tf.float32)
    t = tf.cast(x <= a0, tf.float32)

    lr = s * (x / warmup) + \
         (1 - s) * (t * (x * (a1-1)/a0 + 1) +
                    (1 - t) * ((x - 1) * (a1-a2) / (a0-1) + a2))
    return lr

def warmup_4_1(x, warmup=0.002):
    return warmup_new(x, 9, 4, 1, warmup)

def warmup_6_3(x, warmup=0.002):
    return warmup_new(x, 9, 6, 3, warmup)

def warmup_841(x, warmup=0.002):
    return warmup_new(x, 8, 6, 2, warmup)

def warmup_863(x, warmup=0.002):
    return warmup_new(x, 8, 6, 3, warmup)

def warmup_751(x, warmup=0.002):
    return warmup_new(x, 8, 6, 2, warmup)

def warmup_773(x, warmup=0.002):
    return warmup_new(x, 8, 6, 2, warmup)

def schedules_name2fn(name):
    if name == 'warmup_cosine':
        fn = warmup_cosine
    elif name == 'warmup_constant':
        fn = warmup_constant
    elif name == 'warmup_linear':
        fn = warmup_linear
    elif name == 'warmup_exp':
        fn = warmup_exp
    elif name == 'warmup_exp5':
        fn = warmup_exp5
    elif name == 'warmup_841':
        fn = warmup_841
    elif name == 'warmup_863':
        fn = warmup_863
    elif name == 'warmup_751':
        fn = warmup_841
    elif name == 'warmup_773':
        fn = warmup_841
    else:
        raise NotImplementedError
    return fn