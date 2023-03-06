import paddle
from typing import Optional, List
import math
import copy
from utils.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn
from .param_init import constant_init, xavier_uniform_init, normal_init
from .depth_predictor.transformer import MultiheadAttention


class DepthAwareTransformer(paddle.nn.Layer):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
        num_decoder_layers=6, dim_feedforward=1024, dropout=0.1, activation
        ='relu', return_intermediate_dec=False, num_feature_levels=4,
        dec_n_points=4, enc_n_points=4, two_stage=False,
        two_stage_num_proposals=50):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        encoder_layer = VisualEncoderLayer(d_model, dim_feedforward,
            dropout, activation, num_feature_levels, nhead, enc_n_points)
        self.encoder = VisualEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = DepthAwareDecoderLayer(d_model, dim_feedforward,
            dropout, activation, num_feature_levels, nhead, dec_n_points)
        self.decoder = DepthAwareDecoder(decoder_layer, num_decoder_layers,
            return_intermediate_dec)
        self.level_embed = paddle.create_parameter(shape=[num_feature_levels, d_model], dtype='float32')
        if two_stage:
            self.enc_output = paddle.nn.Linear(in_features=d_model,
                out_features=d_model)
            self.enc_output_norm = paddle.nn.LayerNorm(d_model)
            self.pos_trans = paddle.nn.Linear(in_features=d_model * 2,
                out_features=d_model * 2)
            self.pos_trans_norm = paddle.nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = paddle.nn.Linear(in_features=d_model,
                out_features=2)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.rank() > 1:
                xavier_uniform_init(p)
        for m in self.sublayers():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_init(self.reference_points.weight, gain=1.0)
            constant_init(self.reference_points.bias, value=0.0)
            normal_init(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi
        dim_t = paddle.arange(start=num_pos_feats, dtype='float32')
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        proposals = paddle.nn.functional.sigmoid(proposals) * scale
        pos = proposals[:, :, :, (None)] / dim_t
        pos = paddle.stack(x=(pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].
            cos()), axis=4).flatten(start_axis=2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask,
        spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:_cur + H_ * W_].reshape([N_, H_, W_, 1])
            valid_H = paddle.sum(x=~mask_flatten_[:, :, (0), (0)], axis=1)
            valid_W = paddle.sum(x=~mask_flatten_[:, (0), :, (0)], axis=1)
            grid_y, grid_x = paddle.meshgrid(paddle.linspace(start=0, stop=
                H_ - 1, num=H_, dtype='float32'), paddle.linspace(start=0,
                stop=W_ - 1, num=W_, dtype='float32'))
            grid = paddle.concat(x=[grid_x.unsqueeze(axis=-1), grid_y.
                unsqueeze(axis=-1)], axis=-1)
            scale = paddle.concat(x=[valid_W.unsqueeze(axis=-1), valid_H.
                unsqueeze(axis=-1)], axis=1).reshape([N_, 1, 1, 2])
            grid = (grid.unsqueeze(axis=0).expand([N_, -1, -1, -1]) + 0.5
                ) / scale
            lr = paddle.ones_like(x=grid) * 0.05 * 2.0 ** lvl
            tb = paddle.ones_like(x=grid) * 0.05 * 2.0 ** lvl
            wh = paddle.concat(x=(lr, tb), axis=-1)
            proposal = paddle.concat(x=(grid, wh), axis=-1).reshape([N_, -1, 6])
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = paddle.concat(x=proposals, axis=1)
        output_proposals_valid = ((output_proposals > 0.01) & (
            output_proposals < 0.99)).all(axis=-1, keepdim=True)
        output_proposals = paddle.log(x=output_proposals / (1 -
            output_proposals))
        output_proposals = paddle.where(memory_padding_mask.unsqueeze(axis=-1), paddle.full(output_proposals.shape, float('inf'), output_proposals.dtype), output_proposals)
        output_proposals = paddle.where(~output_proposals_valid, paddle.full(output_proposals.shape, float('inf'), output_proposals.dtype), output_proposals)
        output_memory = memory
        output_memory = paddle.where(memory_padding_mask.unsqueeze(axis=-1), paddle.full(output_memory.shape, float(0), output_memory.dtype), output_memory)
        output_memory = paddle.where(~output_proposals_valid, paddle.full(output_memory.shape, float(0), output_memory.dtype), output_memory)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = paddle.sum(x=~mask[:, :, (0)], axis=1)
        valid_W = paddle.sum(x=~mask[:, (0), :], axis=1)
        valid_ratio_h = valid_H.cast('float32') / H
        valid_ratio_w = valid_W.cast('float32') / W
        valid_ratio = paddle.stack(x=[valid_ratio_w, valid_ratio_h], axis=-1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None,
        depth_pos_embed=None):
        assert self.two_stage or query_embed is not None
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks,
            pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = h, w
            spatial_shapes.append(spatial_shape)
            perm_4 = list(range(len(src.flatten(start_axis=2).shape)))
            perm_4[1] = 2
            perm_4[2] = 1
            src = src.flatten(start_axis=2).transpose(perm=perm_4)
            perm_5 = list(range(len(pos_embed.flatten(start_axis=2).shape)))
            perm_5[1] = 2
            perm_5[2] = 1
            pos_embed = pos_embed.flatten(start_axis=2).transpose(perm=perm_5)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].reshape([1, 1, -1])
            mask = mask.cast('int32').flatten(start_axis=1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = paddle.concat(x=src_flatten, axis=1)
        lvl_pos_embed_flatten = paddle.concat(x=lvl_pos_embed_flatten, axis=1)
        mask_flatten = paddle.concat(x=mask_flatten, axis=1)
        spatial_shapes = paddle.to_tensor(data=spatial_shapes, dtype=
            'int64', place=srcs[0].place)
        """Class Method: *.new_zeros, not convert, please check which one of torch.Tensor.*/Optimizer.*/nn.Module.* it is, and convert manually"""
        level_start_index = paddle.concat(x=(paddle.zeros((1,), dtype=spatial_shapes.dtype),
            spatial_shapes.prod(axis=1).cumsum(axis=0)[:-1]))
        valid_ratios = paddle.stack(x=[self.get_valid_ratio(m) for m in masks], axis=1)
        memory = self.encoder(src_flatten, spatial_shapes,
            level_start_index, valid_ratios, lvl_pos_embed_flatten,
            mask_flatten)
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = (self.
                gen_encoder_output_proposals(memory, mask_flatten,
                spatial_shapes))
            enc_outputs_class = self.decoder.class_embed[self.decoder.
                num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.
                num_layers](output_memory) + output_proposals
            topk = self.two_stage_num_proposals
            topk_proposals = paddle.topk(x=enc_outputs_class[..., 0], k=
                topk, axis=1)[1]
            topk_coords_unact = paddle.take_along_axis(arr=
                enc_outputs_coord_unact, axis=1, indices=topk_proposals.
                unsqueeze(axis=-1).tile(repeat_times=[1, 1, 6]))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = paddle.nn.functional.sigmoid(topk_coords_unact)
            init_reference_out = reference_points
            topk_coords_unact_input = paddle.concat(x=(topk_coords_unact[(
                ...), 0:2], topk_coords_unact[(...), 2::2] +
                topk_coords_unact[(...), 3::2]), axis=-1)
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.
                get_proposal_pos_embed(topk_coords_unact_input)))
            query_embed, tgt = paddle.split(pos_trans_out, num_or_sections=[c, c], axis=2)
        else:
            query_embed, tgt = paddle.split(query_embed, num_or_sections=[c, c], axis=1)
            query_embed = query_embed.unsqueeze(axis=0).expand([bs, -1, -1])
            tgt = tgt.unsqueeze(axis=0).expand([bs, -1, -1])
            reference_points = paddle.nn.functional.sigmoid(self.reference_points(query_embed))
            init_reference_out = reference_points
        depth_pos_embed = depth_pos_embed.flatten(start_axis=2).transpose(perm
            =[2, 0, 1])
        mask_depth = masks[1].cast('int32').flatten(start_axis=1)
        hs, inter_references, inter_references_dim = self.decoder(tgt,
            reference_points, memory, spatial_shapes, level_start_index,
            valid_ratios, query_embed, mask_flatten, depth_pos_embed,
            mask_depth)
        inter_references_out = inter_references
        inter_references_out_dim = inter_references_dim
        if self.two_stage:
            return (hs, init_reference_out, inter_references_out,
                inter_references_out_dim, enc_outputs_class,
                enc_outputs_coord_unact)
        return (hs, init_reference_out, inter_references_out,
            inter_references_out_dim, None, None)


class VisualEncoderLayer(paddle.nn.Layer):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation=
        'relu', n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = paddle.nn.Dropout(p=dropout)
        self.norm1 = paddle.nn.LayerNorm(d_model)
        self.linear1 = paddle.nn.Linear(in_features=d_model, out_features=d_ffn
            )
        self.activation = _get_activation_fn(activation)
        self.dropout2 = paddle.nn.Dropout(p=dropout)
        self.linear2 = paddle.nn.Linear(in_features=d_ffn, out_features=d_model
            )
        self.dropout3 = paddle.nn.Dropout(p=dropout)
        self.norm2 = paddle.nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes,
        level_start_index, padding_mask=None):
        src2 = self.self_attn(self.with_pos_embed(src, pos),
            reference_points, src, spatial_shapes, level_start_index,
            padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = self.forward_ffn(src)
        return src


class VisualEncoder(paddle.nn.Layer):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = paddle.meshgrid(paddle.linspace(start=0.5, stop=
                H_ - 0.5, num=H_, dtype='float32'), paddle.linspace(start=
                0.5, stop=W_ - 0.5, num=W_, dtype='float32'))
            ref_y = ref_y.reshape([-1])[None] / (valid_ratios[:, (None), (lvl), (1)] * H_)
            ref_x = ref_x.reshape([-1])[None] / (valid_ratios[:, (None), (lvl), (0)] * W_)
            ref = paddle.stack(x=(ref_x, ref_y), axis=-1)
            reference_points_list.append(ref)
        reference_points = paddle.concat(x=reference_points_list, axis=1)
        reference_points = reference_points[:, :, (None)] * valid_ratios[:,
            (None)]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios,
        pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes,
            valid_ratios, device=src.place)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes,
                level_start_index, padding_mask)
        return output


class DepthAwareDecoderLayer(paddle.nn.Layer):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation=
        'relu', n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = paddle.nn.Dropout(p=dropout)
        self.norm1 = paddle.nn.LayerNorm(d_model)
        self.cross_attn_depth = MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_depth = paddle.nn.Dropout(p=dropout)
        self.norm_depth = paddle.nn.LayerNorm(d_model)
        self.self_attn = MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = paddle.nn.Dropout(p=dropout)
        self.norm2 = paddle.nn.LayerNorm(d_model)
        self.linear1 = paddle.nn.Linear(in_features=d_model, out_features=d_ffn
            )
        self.activation = _get_activation_fn(activation)
        self.dropout3 = paddle.nn.Dropout(p=dropout)
        self.linear2 = paddle.nn.Linear(in_features=d_ffn, out_features=d_model
            )
        self.dropout4 = paddle.nn.Dropout(p=dropout)
        self.norm3 = paddle.nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src,
        src_spatial_shapes, level_start_index, src_padding_mask,
        depth_pos_embed, mask_depth):
        mask_depth = mask_depth.unsqueeze([1, 2])
        tgt2 = self.cross_attn_depth(tgt.transpose(perm=[1, 0, 2]),
            depth_pos_embed, depth_pos_embed, key_padding_mask=mask_depth).transpose(perm=[1, 0, 2])
        tgt = tgt + self.dropout_depth(tgt2)
        tgt = self.norm_depth(tgt)
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(perm=[1,0,2]), k.transpose(perm=
            [1,0,2]), tgt.transpose(perm=[1,0,2])).transpose(perm=[1,0,2])
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
            reference_points, src, src_spatial_shapes, level_start_index,
            src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self.forward_ffn(tgt)
        return tgt


class DepthAwareDecoder(paddle.nn.Layer):

    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.dim_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes,
        src_level_start_index, src_valid_ratios, query_pos=None,
        src_padding_mask=None, depth_pos_embed=None, mask_depth=None):
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        intermediate_reference_dims = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 6:
                reference_points_input = reference_points[:, :, (None)
                    ] * paddle.concat(x=[src_valid_ratios, src_valid_ratios,
                    src_valid_ratios], axis=-1)[:, (None)]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, (None)
                    ] * src_valid_ratios[:, (None)]
            output = layer(output, query_pos, reference_points_input, src,
                src_spatial_shapes, src_level_start_index, src_padding_mask,
                depth_pos_embed, mask_depth)
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 6:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points)
                    new_reference_points = paddle.nn.functional.sigmoid(new_reference_points)
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[(...), :2] = tmp[(...), :2
                        ] + inverse_sigmoid(reference_points)
                    new_reference_points = paddle.nn.functional.sigmoid(new_reference_points)
                reference_points = new_reference_points.detach()
            if self.dim_embed is not None:
                reference_dims = self.dim_embed[lid](output)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_reference_dims.append(reference_dims)
        if self.return_intermediate:
            return paddle.stack(x=intermediate), paddle.stack(x=
                intermediate_reference_points), paddle.stack(x=
                intermediate_reference_dims)
        return output, reference_points


def _get_clones(module, N):
    return paddle.nn.LayerList(sublayers=[copy.deepcopy(module) for i in
        range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == 'relu':
        return paddle.nn.functional.relu
    if activation == 'gelu':
        return paddle.nn.functional.gelu
    if activation == 'glu':
        return paddle.nn.functional.glu
    raise RuntimeError(f'activation should be relu/gelu, not {activation}.')


def build_depthaware_transformer(cfg):
    return DepthAwareTransformer(d_model=cfg['hidden_dim'], dropout=cfg[
        'dropout'], activation='relu', nhead=cfg['nheads'], dim_feedforward
        =cfg['dim_feedforward'], num_encoder_layers=cfg['enc_layers'],
        num_decoder_layers=cfg['dec_layers'], return_intermediate_dec=cfg[
        'return_intermediate_dec'], num_feature_levels=cfg[
        'num_feature_levels'], dec_n_points=cfg['dec_n_points'],
        enc_n_points=cfg['enc_n_points'], two_stage=cfg['two_stage'],
        two_stage_num_proposals=cfg['num_queries'])
