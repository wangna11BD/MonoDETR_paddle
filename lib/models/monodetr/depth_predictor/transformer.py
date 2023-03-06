import paddle
import copy

class MultiheadAttention(paddle.nn.Layer):
    """
    """
    def __init__(self,
                 embed_dims,
                 num_heads,
                 dropout=0.0,
                 batch_first=False,
                 **kwargs):
        super(MultiheadAttention, self).__init__()

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = paddle.nn.MultiHeadAttention(embed_dims, num_heads, dropout,
                                          **kwargs)

    def forward(self,
                query,
                key=None,
                value=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):

        if not self.batch_first:
            query = query.transpose([1, 0, 2])
            key = key.transpose([1, 0, 2])
            value = value.transpose([1, 0, 2])

        if key_padding_mask is None:
            out = self.attn(
                query=query, key=key, value=value, attn_mask=attn_mask)
        else:
            if attn_mask is None:
                attn_mask = ~key_padding_mask
                out = self.attn(
                    query=query, key=key, value=value, attn_mask=attn_mask.cast('bool'))
            else:
                raise ValueError('key_padding_mask is not None')

        if not self.batch_first:
            out = out.transpose([1, 0, 2])

        return out


def _get_clones(module, N):
    return paddle.nn.LayerList(sublayers=[copy.deepcopy(module) for i in
        range(N)])


class TransformerEncoder(paddle.nn.Layer):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_key_padding_mask, pos):
        output = src
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(paddle.nn.Layer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
        activation='relu'):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead,
            dropout=dropout)
        self.linear1 = paddle.nn.Linear(in_features=d_model, out_features=dim_feedforward)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.linear2 = paddle.nn.Linear(in_features=dim_feedforward,
            out_features=d_model)
        self.norm1 = paddle.nn.LayerNorm(d_model)
        self.norm2 = paddle.nn.LayerNorm(d_model)
        self.dropout1 = paddle.nn.Dropout(p=dropout)
        self.dropout2 = paddle.nn.Dropout(p=dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_key_padding_mask, pos):
        q = k = self.with_pos_embed(src, pos)
        src_key_padding_mask = src_key_padding_mask.unsqueeze([1, 2])
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return paddle.nn.functional.relu
    if activation == 'gelu':
        return paddle.nn.functional.gelu
    if activation == 'glu':
        return paddle.nn.functional.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')
