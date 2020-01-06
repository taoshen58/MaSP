import tensorflow as tf
from peach.tf_nn.attn import compatibility_fn, direct_mask_generation, split_head, combine_head
from peach.tf_nn.nn import residual_connection, residual_connection_with_dense, bn_dense_layer_v2, dropout, \
    bn_dense_layer_multi_head
from peach.tf_nn.general import exp_mask_v3, mask_v3, act_name2fn, get_shape_list


def transformer_seq_decoder(
        dec_input_emb_mat, decoder_ids, encoder_states, decoder_mask, encoder_mask, n_out_channel, num_layers,
        decoder_history_inputs=None,
        hn=768, head_num=12, act_name="gelu",
        wd=0., is_training=None, keep_prob_dense=1., keep_prob_attn=1., keep_prob_res=1.,
        scope=None):
    with tf.variable_scope(scope or "transformer_seq_decoder"):
        with tf.variable_scope("decoder_emb"):
            decoder_inputs = tf.nn.embedding_lookup(dec_input_emb_mat, decoder_ids)  # bs,sl,hn

        with tf.variable_scope("decoder_recurrence"):
            dec_outputs, new_decoder_history_inputs = transformer_decoder(  # bs,sl,hn
                decoder_inputs, encoder_states, decoder_mask, encoder_mask, num_layers,
                decoder_history_inputs,
                hn, head_num, act_name,
                wd, is_training, keep_prob_dense, keep_prob_attn, keep_prob_res,
                scope="transformer_decoder"
            )
            # prediction logits: two layer
            # pre_logits_seq2seq = bn_dense_layer_v2(
            #     dec_outputs, hn, True, 0., "pre_logits_seq2seq", act_name,
            #     False, 0., keep_prob_dense, is_training
            # )
            logits_seq2seq = bn_dense_layer_v2(  # bs,sl,
                dec_outputs, n_out_channel, True, 0., "logits_seq2seq", "linear",
                False, 0., keep_prob_dense, is_training
            )
            return dec_outputs, logits_seq2seq, new_decoder_history_inputs


def transformer_decoder(
        decoder_input, encoder_output,
        decoder_mask, encoder_mask,
        num_layers, decoder_history_inputs=None,
        hn=768, head_num=12, act_name="gelu",
        wd=0., is_training=None, keep_prob_dense=1., keep_prob_attn=1., keep_prob_res=1.,
        scope=None,

):
    fwd_mask = direct_mask_generation(decoder_mask, direct="forward", attn_self=True)  # DONE: double check this

    use_decoder_history = False
    decoder_history_inputs_list = []
    if not isinstance(decoder_history_inputs, type(None)):
        use_decoder_history = True
        decoder_history_inputs_list = tf.unstack(decoder_history_inputs, num_layers, axis=1)
        fwd_mask = None
    cur_history_inputs_list = []

    with tf.variable_scope(scope or "transformer_decoder"):
        x = decoder_input

        for layer in range(num_layers):
            with tf.variable_scope("layer_{}".format(layer)):
                tensor_to_prev = decoder_history_inputs_list[layer] if use_decoder_history else None
                cur_history_inputs_list.append(x)
                with tf.variable_scope("self_attention"):
                    y = multihead_attention_decoder(
                        x, x, decoder_mask, fwd_mask, "linear",
                        hn, head_num, wd, is_training, keep_prob_dense, keep_prob_attn,
                        tensor_to_prev=tensor_to_prev, mask_prev_to=None,
                    )

                    x = residual_connection(x, y, is_training, keep_prob_res)
                with tf.variable_scope("encdec_attention"):
                    y = multihead_attention_decoder(
                        x, encoder_output, encoder_mask, None, "linear",
                        hn, head_num, wd, is_training, keep_prob_dense, keep_prob_attn,
                    )

                    x = residual_connection(x, y, is_training, keep_prob_res)
                with tf.variable_scope("ffn"):
                    x = residual_connection_with_dense(
                        x, 4 * hn, True, 0., "res_ffn", act_name, False,
                        wd, keep_prob_dense, is_training, keep_prob_res)

    new_decoder_history_inputs = None
    if use_decoder_history:
        cur_history_inputs = tf.stack(cur_history_inputs_list, axis=1)  # bs,num_layer,1,hn
        new_decoder_history_inputs = tf.concat([decoder_history_inputs, cur_history_inputs], axis=2)
    return x, new_decoder_history_inputs


def multihead_attention_decoder(
        tensor_from, tensor_to, mask_to, mask_direction=None,  # [bs,slf,slt]
        act_name="relu", hn=768, head_num=12, wd=0., is_training=None, keep_prob_dense=1., keep_prob_attn=1.,
        tensor_to_prev=None, mask_prev_to=None,
        scope=None,
):
    head_dim = hn // head_num
    with tf.variable_scope(scope or "multihead_attention_decoder"):
        # if not isinstance(tensor_to_prev, type(None)):  # to print the shape
        #     tensor_from = tf.Print(tensor_from, [
        #         tf.shape(tensor_from), tf.shape(tensor_to),  tf.shape(mask_to),  tf.shape(tensor_to_prev)])

        if isinstance(tensor_to_prev, type(None)):
            tensor_to_all = tensor_to # bs,sl,hn
            mask_to_all = mask_to  # bs,sl
        else:
            tensor_to_all = tf.concat([tensor_to_prev, tensor_to], -2)  # bs,psl+1,hn
            if mask_prev_to is None:
                mask_prev_to = tf.cast(tf.ones(get_shape_list(tensor_to_prev, 3)[:2] , tf.int32), tf.bool)  # bs,psl
            mask_to_all = tf.concat([mask_prev_to, mask_to], -1)  # bs,psl+1

        attn_scores = compatibility_fn(
            tensor_from, tensor_to_all, method="multi_head", head_num=head_num,
            hn=hn, wd=wd, is_training=is_training, keep_prob=keep_prob_dense,
        )  # [bs,hd_num,slf,slt]
        v_heads = bn_dense_layer_v2(  # bs,slt,hd_dim * hd_num
            tensor_to_all, head_dim, True, 0., 'v_heads',
            'linear', False, wd, keep_prob_dense, is_training, dup_num=head_num
        )
        v_heads = split_head(v_heads, head_num)  # # bs,hd_num,slt,hd_dim

        # mask the self-attention scores
        attn_scores_mask = tf.expand_dims(mask_to_all, 1)  # bs,1,tsl
        if (not isinstance(mask_direction, type(None))) and isinstance(tensor_to_prev, type(None)):
            attn_scores_mask = tf.logical_and(attn_scores_mask, mask_direction)  # bs,tsl,tsl
        attn_scores_masked = exp_mask_v3(attn_scores, attn_scores_mask, multi_head=True)  # [bs,hd_num,slf,slt]
        attn_prob = tf.nn.softmax(attn_scores_masked)
        attn_prob = dropout(attn_prob, keep_prob_attn, is_training)  # [bs,hd_num,slf,slt]

        v_heads_etd = tf.expand_dims(v_heads, 2)  # bs,hd_num,1,slt,hd_dim
        attn_prob_etd = tf.expand_dims(attn_prob, -1)  # bs,hd_num,slf,slt,1

        attn_res = tf.reduce_sum(v_heads_etd * attn_prob_etd, 3)  # bs,hd_num,slf,hd_dim
        out_prev = combine_head(attn_res)  # bs,fsl,hn

        # if mask_direction is not None and tensor_to_prev is None:
        #     attn_scores = exp_mask_v3(attn_scores, mask_direction, multi_head=True)  # [bs,hd_num,slf,slt]
        # attn_scores = dropout(attn_scores, keep_prob_attn, is_training)
        #
        # attn_res = softsel( # [bs,hd_num,slf,dhn]
        #     v_heads, attn_scores, mask_to_all,
        #     mask_add_head_dim_for_scores=True,
        #     input_add_multi_head_dim=False,
        #     score_add_hn_dim=True,
        #     axis=3)
        # out_prev = combine_head(attn_res)
        # dense layer
        out = bn_dense_layer_v2(
            out_prev, hn, True, 0., "output_transformer", act_name, False, wd, keep_prob_dense, is_training
        )
        return out


# self-attention
def s2t_self_attn(  # compatible with lower version of tensorflow
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
        attn_prob = tf.nn.softmax(align_scores_masked, dim=len(get_shape_list(align_scores_masked))-2)  # bs,sl,hn

        if 'attn_keep_prob' in kwargs and isinstance(kwargs['attn_keep_prob'], float):
            attn_prob = dropout(attn_prob, kwargs['attn_keep_prob'], is_training)  # bs,sl,hn

        attn_res = tf.reduce_sum(  # [bs,sl,hn] -> [bs,dim]
            mask_v3(attn_prob*tensor_input, tensor_mask, high_dim=True), axis=-2
        )

        return attn_res  # [bs,hn]

