import paddle
"""
MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection
"""
import math
import copy
from utils import box_ops
from utils.misc import NestedTensor, nested_tensor_from_tensor_list, accuracy, get_world_size, is_dist_avail_and_initialized, inverse_sigmoid
from .backbone import build_backbone
from .matcher import build_matcher
from .depthaware_transformer import build_depthaware_transformer
from .depth_predictor import DepthPredictor
from .depth_predictor.ddn_loss import DDNLoss
from lib.losses.focal_loss import sigmoid_focal_loss
from .param_init import constant_init, xavier_uniform_init


def _get_clones(module, N):
    return paddle.nn.LayerList(sublayers=[copy.deepcopy(module) for i in
        range(N)])


class MonoDETR(paddle.nn.Layer):
    """ This is the MonoDETR module that performs monocualr 3D object detection """

    def __init__(self, backbone, depthaware_transformer, depth_predictor,
        num_classes, num_queries, num_feature_levels, aux_loss=True,
        with_box_refine=False, two_stage=False, init_box=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            depthaware_transformer: depth-aware transformer architecture. See depth_aware_transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For KITTI, we recommend 50 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage MonoDETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.depthaware_transformer = depthaware_transformer
        self.depth_predictor = depth_predictor
        hidden_dim = depthaware_transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.class_embed = paddle.nn.Linear(in_features=hidden_dim,
            out_features=num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = paddle.ones(shape=[num_classes]
            ) * bias_value
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2)
        if init_box == True:
            constant_init(self.bbox_embed.layers[-1].weight, value=0.0)
            constant_init(self.bbox_embed.layers[-1].bias, value=0.0)
        if not two_stage:
            self.query_embed = paddle.nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=in_channels, 
                                                            out_channels=hidden_dim, kernel_size=1), 
                                                            paddle.nn.GroupNorm(32, hidden_dim)))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(paddle.nn.Sequential(paddle.nn.
                    Conv2D(in_channels=in_channels, out_channels=hidden_dim,
                    kernel_size=3, stride=2, padding=1), 
                    paddle.nn.GroupNorm(32, hidden_dim)))
                in_channels = hidden_dim
            self.input_proj = paddle.nn.LayerList(sublayers=input_proj_list)
        else:
            self.input_proj = paddle.nn.LayerList(sublayers=[paddle.nn.
                Sequential(paddle.nn.Conv2D(in_channels=backbone.
                num_channels[0], out_channels=hidden_dim, kernel_size=1),
                paddle.nn.GroupNormm(32, hidden_dim))])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        for proj in self.input_proj:
            xavier_uniform_init(proj[0].weight, gain=1)
            constant_init(proj[0].bias, value=0)
        num_pred = (depthaware_transformer.decoder.num_layers + 1 if
            two_stage else depthaware_transformer.decoder.num_layers)
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            constant_init(self.bbox_embed[0].layers[-1].bias[2:], value=-2.0)
            self.depthaware_transformer.decoder.bbox_embed = self.bbox_embed
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.depthaware_transformer.decoder.dim_embed = self.dim_embed_3d
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.depth_embed = _get_clones(self.depth_embed, num_pred)
        else:
            constant_init(self.bbox_embed.layers[-1].bias[2:], value=-2.0)
            self.class_embed = paddle.nn.LayerList(sublayers=[self.
                class_embed for _ in range(num_pred)])
            self.bbox_embed = paddle.nn.LayerList(sublayers=[self.
                bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = paddle.nn.LayerList(sublayers=[self.
                dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = paddle.nn.LayerList(sublayers=[self.
                angle_embed for _ in range(num_pred)])
            self.depth_embed = paddle.nn.LayerList(sublayers=[self.
                depth_embed for _ in range(num_pred)])
            self.depthaware_transformer.decoder.bbox_embed = None
        if two_stage:
            self.depthaware_transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                constant_init(box_embed.layers[-1].bias[2:], value=0.0)

    def forward(self, images, calibs, targets, img_sizes):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        features, pos = self.backbone(images)
        
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = paddle.zeros(shape=[src.shape[0], src.shape[2], src.shape[3]], dtype='int64')
                mask = paddle.nn.functional.interpolate(x=m[None].cast(dtype='float32'),
                    size=src.shape[-2:]).cast(dtype='bool')[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).cast(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        pred_depth_map_logits, depth_pos_embed, weighted_depth = (self.depth_predictor(srcs, masks[1], pos[1]))
        (hs, init_reference, inter_references, inter_references_dim,
            enc_outputs_class, enc_outputs_coord_unact) = (self.
            depthaware_transformer(srcs, masks, pos, query_embeds,
            depth_pos_embed))
        outputs_coords = []
        outputs_classes = []
        outputs_3d_dims = []
        outputs_depths = []
        outputs_angles = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[(...), :2] += reference
            outputs_coord = paddle.nn.functional.sigmoid(tmp)
            outputs_coords.append(outputs_coord)
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)
            size3d = inter_references_dim[lvl]
            outputs_3d_dims.append(size3d)
            box2d_height_norm = outputs_coord[:, :, (4)] + outputs_coord[:,
                :, (5)]
            box2d_height = paddle.clip(x=box2d_height_norm * img_sizes[:, 1
                :2], min=1.0)
            depth_geo = size3d[:, :, (0)] / box2d_height * calibs[:, (0), (0)
                ].unsqueeze(axis=1)
            depth_reg = self.depth_embed[lvl](hs[lvl])
            outputs_center3d = ((outputs_coord[(...), :2] - 0.5) * 2
                ).unsqueeze(axis=2).detach()
            depth_map = paddle.nn.functional.grid_sample(x=weighted_depth.
                unsqueeze(axis=1), grid=outputs_center3d, mode='bilinear',
                align_corners=True).squeeze(axis=1)
            depth_ave = paddle.concat(x=[(1.0 / (paddle.nn.functional.sigmoid(depth_reg[:, :, 0:1]) + \
                1e-06) - 1.0 + depth_geo.unsqueeze(axis=-1) +
                depth_map) / 3, depth_reg[:, :, 1:2]], axis=-1)
            outputs_depths.append(depth_ave)
            outputs_angle = self.angle_embed[lvl](hs[lvl])
            outputs_angles.append(outputs_angle)
        outputs_coord = paddle.stack(x=outputs_coords)
        outputs_class = paddle.stack(x=outputs_classes)
        outputs_3d_dim = paddle.stack(x=outputs_3d_dims)
        outputs_depth = paddle.stack(x=outputs_depths)
        outputs_angle = paddle.stack(x=outputs_angles)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes':
            outputs_coord[-1]}
        out['pred_3d_dim'] = outputs_3d_dim[-1]
        out['pred_depth'] = outputs_depth[-1]
        out['pred_angle'] = outputs_angle[-1]
        out['pred_depth_map_logits'] = pred_depth_map_logits
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class,
                outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth)
        if self.two_stage:
            enc_outputs_coord = paddle.nn.functional.sigmoid(enc_outputs_coord_unact)
            out['enc_outputs'] = {'pred_logits': enc_outputs_class,
                'pred_boxes': enc_outputs_coord}
        return out

    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim,
        outputs_angle, outputs_depth):
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_3d_dim': c,
            'pred_angle': d, 'pred_depth': e} for a, b, c, d, e in zip(
            outputs_class[:-1], outputs_coord[:-1], outputs_3d_dim[:-1],
            outputs_angle[:-1], outputs_depth[:-1])]


class SetCriterion(paddle.nn.Layer):
    """ This class computes the loss for MonoDETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss()

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = []
        for t, (_, J) in zip(targets, indices):
            if J.shape[0] != 0:
                target_classes_o.append(t['labels'][J])
        target_classes_o = paddle.concat(target_classes_o)
        # target_classes_o = paddle.concat(x=[t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = paddle.full(shape=src_logits.shape[:2], fill_value=self.num_classes, dtype='int64')
        target_classes[idx] = target_classes_o.squeeze().cast(dtype='int64')
        target_classes_onehot = paddle.zeros(shape=[src_logits.shape[0],
            src_logits.shape[1], src_logits.shape[2] + 1], dtype=src_logits.dtype)
        target_classes_onehot = paddle.put_along_axis(target_classes_onehot, 
            indices=target_classes.unsqueeze(axis=-1), values=1.0, axis=2)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot,
            num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx],
                target_classes_o)[0]
        return losses

    @paddle.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        tgt_lengths = paddle.to_tensor(data=[len(v['labels']) for v in
            targets])
        card_pred = (pred_logits.argmax(axis=-1) != pred_logits.shape[-1] - 1
            ).sum(axis=1)
        card_err = paddle.nn.functional.l1_loss(card_pred.cast('float32'),
            tgt_lengths.cast('float32'))
        losses = {'cardinality_error': card_err}
        return losses

    def loss_3dcenter(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        if idx[1].shape[0]==1:
            src_3dcenter = outputs['pred_boxes'][:, :, 0:2][idx].unsqueeze(0)
        else:
            src_3dcenter = outputs['pred_boxes'][:, :, 0:2][idx]

        target_3dcenter = []
        for t, (_, i) in zip(targets, indices):
            if i.shape[0] != 0 and t['boxes_3d'].shape[0] != 0:
                if i.shape[0] == 1:
                    target_3dcenter.append(t['boxes_3d'][:, 0:2][i].unsqueeze(0))
                else:
                    target_3dcenter.append(t['boxes_3d'][:, 0:2][i])
        target_3dcenter = paddle.concat(target_3dcenter, axis=0)

        # target_3dcenter = paddle.concat(x=[t['boxes_3d'][:, 0:2][i] for t,
        #     (_, i) in zip(targets, indices)], axis=0)
        loss_3dcenter = paddle.nn.functional.l1_loss(src_3dcenter,
            target_3dcenter, reduction='none')
        losses = {}
        losses['loss_center'] = loss_3dcenter.sum() / num_boxes
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_2dboxes = outputs['pred_boxes'][:, :, 2:6][idx]
        target_2dboxes = []
        for t, (_, i) in zip(targets, indices):
            if i.shape[0] != 0 and t['boxes_3d'].shape[0] != 0:
                if i.shape[0] == 1:
                    target_2dboxes.append(t['boxes_3d'][:, 2:6][i].unsqueeze(0))
                else:
                    target_2dboxes.append(t['boxes_3d'][:, 2:6][i])
        target_2dboxes = paddle.concat(target_2dboxes, axis=0)
        # target_2dboxes = paddle.concat(x=[t['boxes_3d'][:, 2:6][i] for t, (
        #     _, i) in zip(targets, indices)], axis=0)
        loss_bbox = paddle.nn.functional.l1_loss(src_2dboxes, target_2dboxes,
            reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        if idx[1].shape[0]==1:
            src_boxes = outputs['pred_boxes'][idx].unsqueeze(0)
        else:
            src_boxes = outputs['pred_boxes'][idx]
        target_boxes = []
        for t, (_, i) in zip(targets, indices):
            if i.shape[0] != 0 and t['boxes_3d'].shape[0] != 0:
                if i.shape[0] == 1:
                    target_boxes.append(t['boxes_3d'][i].unsqueeze(0))
                else:
                    target_boxes.append(t['boxes_3d'][i])
        target_boxes = paddle.concat(target_boxes, axis=0)
        # target_boxes = paddle.concat(x=[t['boxes_3d'][i] for t, (_, i) in
        #     zip(targets, indices)], axis=0)
        loss_giou = 1 - paddle.diag(x=box_ops.generalized_box_iou(box_ops.
            box_cxcylrtb_to_xyxy(src_boxes), box_ops.box_cxcylrtb_to_xyxy(
            target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_depths(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        if idx[1].shape[0]==1:
            src_depths = outputs['pred_depth'][idx].unsqueeze(0)
        else:
            src_depths = outputs['pred_depth'][idx]

        target_depths = []
        for t, (_, i) in zip(targets, indices):
            if i.shape[0] != 0 and t['depth'].shape[0] != 0:
                if i.shape[0] == 1:
                    target_depths.append(t['depth'][i].unsqueeze(0))
                else:
                    target_depths.append(t['depth'][i])
        target_depths = paddle.concat(target_depths, axis=0).squeeze()

        # target_depths = paddle.concat(x=[t['depth'][i] for t, (_, i) in zip
        #     (targets, indices)], axis=0).squeeze()
        depth_input, depth_log_variance = src_depths[:, (0)], src_depths[:, (1)
            ]
        depth_loss = 1.4142 * paddle.exp(x=-depth_log_variance) * paddle.abs(x
            =depth_input - target_depths) + depth_log_variance
        losses = {}
        losses['loss_depth'] = depth_loss.sum() / num_boxes
        return losses

    def loss_dims(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        if idx[1].shape[0]==1:
            src_dims = outputs['pred_3d_dim'][idx].unsqueeze(0)
        else:
            src_dims = outputs['pred_3d_dim'][idx]

        target_dims = []
        for t, (_, i) in zip(targets, indices):
            if i.shape[0] != 0 and t['size_3d'].shape[0] != 0:
                if i.shape[0] == 1:
                    target_dims.append(t['size_3d'][i].unsqueeze(0))
                else:
                    target_dims.append(t['size_3d'][i])
        target_dims = paddle.concat(target_dims, axis=0)

        # target_dims = paddle.concat(x=[t['size_3d'][i] for t, (_, i) in zip
        #     (targets, indices)], axis=0)
        dimension = target_dims.clone().detach()
        dim_loss = paddle.abs(x=src_dims - target_dims)
        dim_loss /= dimension
        with paddle.no_grad():
            compensation_weight = paddle.nn.functional.l1_loss(src_dims,
                target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses['loss_dim'] = dim_loss.sum() / num_boxes
        return losses

    def loss_angles(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        if idx[1].shape[0]==1:
            heading_input = outputs['pred_angle'][idx].unsqueeze(0)
        else:
            heading_input = outputs['pred_angle'][idx]
        
        target_heading_cls = []
        for t, (_, i) in zip(targets, indices):
            if i.shape[0] != 0 and t['heading_bin'].shape[0] != 0:
                if i.shape[0] == 1:
                    target_heading_cls.append(t['heading_bin'][i].unsqueeze(0))
                else:
                    target_heading_cls.append(t['heading_bin'][i])
        target_heading_cls = paddle.concat(target_heading_cls, axis=0)

        target_heading_res = []
        for t, (_, i) in zip(targets, indices):
            if i.shape[0] != 0 and t['heading_res'].shape[0] != 0:
                if i.shape[0] == 1:
                    target_heading_res.append(t['heading_res'][i].unsqueeze(0))
                else:
                    target_heading_res.append(t['heading_res'][i])
        target_heading_res = paddle.concat(target_heading_res, axis=0)

        # target_heading_cls = paddle.concat(x=[t['heading_bin'][i] for t, (_,
        #     i) in zip(targets, indices)], axis=0)
        # target_heading_res = paddle.concat(x=[t['heading_res'][i] for t, (_,
        #     i) in zip(targets, indices)], axis=0)
        heading_input = heading_input.reshape([-1, 24])
        heading_target_cls = target_heading_cls.reshape([-1]).astype(dtype='int64')
        heading_target_res = target_heading_res.reshape([-1])
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = paddle.nn.functional.cross_entropy(input=
            heading_input_cls, label=heading_target_cls, reduction='none')
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = paddle.put_along_axis(paddle.zeros(shape=[heading_target_cls.shape[0], 12]),
                        indices=heading_target_cls.reshape([-1, 1]),
                        values=1.0,
                        axis=1)
        heading_input_res = paddle.sum(x=heading_input_res * cls_onehot, axis=1)
        reg_loss = paddle.nn.functional.l1_loss(heading_input_res,
            heading_target_res, reduction='none')
        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / num_boxes
        return losses

    def loss_depth_map(self, outputs, targets, indices, num_boxes):

        depth_map_logits = outputs['pred_depth_map_logits']
        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = paddle.concat(x=[t['boxes'] for t in targets], 
                    axis=0) * paddle.to_tensor([80, 24, 80, 24])
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = paddle.concat(x=[t['depth'] for t in targets], axis=0
            ).squeeze(axis=1)
        losses = dict()
        losses['loss_depth_map'] = self.ddn_loss(depth_map_logits,
            gt_boxes2d, num_gt_per_img, gt_center_depth)
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = paddle.concat(x=[paddle.full_like(x=src, fill_value=i) for
            i, (src, _) in enumerate(indices)])
        src_idx = paddle.concat(x=[src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = paddle.concat(x=[paddle.full_like(x=tgt, fill_value=i) for
            i, (_, tgt) in enumerate(indices)])
        tgt_idx = paddle.concat(x=[tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {'labels': self.loss_labels, 'cardinality': self.
            loss_cardinality, 'boxes': self.loss_boxes, 'depths': self.
            loss_depths, 'dims': self.loss_dims, 'angles': self.loss_angles,
            'center': self.loss_3dcenter, 'depth_map': self.loss_depth_map}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k !=
            'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = paddle.to_tensor(data=[num_boxes], dtype='float32',
            place=next(iter(outputs.values())).place)
        if is_dist_avail_and_initialized():
            paddle.distributed.all_reduce(num_boxes)
        num_boxes = paddle.clip(x=num_boxes / get_world_size(), min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices,
                num_boxes))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'depth_map':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets,
                        indices, num_boxes, **kwargs)
                    l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class MLP(paddle.nn.Layer):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = paddle.nn.LayerList(sublayers=(paddle.nn.Linear(
            in_features=n, out_features=k) for n, k in zip([input_dim] + h,
            h + [output_dim])))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = paddle.nn.functional.relu(x=layer(x)
                ) if i < self.num_layers - 1 else layer(x)
        return x


def build(cfg):
    backbone = build_backbone(cfg)
    depthaware_transformer = build_depthaware_transformer(cfg)
    depth_predictor = DepthPredictor(cfg)
    model = MonoDETR(backbone, depthaware_transformer, depth_predictor,
        num_classes=cfg['num_classes'], num_queries=cfg['num_queries'],
        aux_loss=cfg['aux_loss'], num_feature_levels=cfg[
        'num_feature_levels'], with_box_refine=cfg['with_box_refine'],
        two_stage=cfg['two_stage'], init_box=cfg['init_box'])
    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg['cls_loss_coef'], 'loss_bbox': cfg[
        'bbox_loss_coef']}
    weight_dict['loss_giou'] = cfg['giou_loss_coef']
    weight_dict['loss_dim'] = cfg['dim_loss_coef']
    weight_dict['loss_angle'] = cfg['angle_loss_coef']
    weight_dict['loss_depth'] = cfg['depth_loss_coef']
    weight_dict['loss_center'] = cfg['3dcenter_loss_coef']
    weight_dict['loss_depth_map'] = cfg['depth_map_loss_coef']
    if cfg['aux_loss']:
        aux_weight_dict = {}
        for i in range(cfg['dec_layers'] - 1):
            aux_weight_dict.update({(k + f'_{i}'): v for k, v in
                weight_dict.items()})
        aux_weight_dict.update({(k + f'_enc'): v for k, v in weight_dict.
            items()})
        weight_dict.update(aux_weight_dict)
    losses = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles',
        'center', 'depth_map']
    criterion = SetCriterion(cfg['num_classes'], matcher=matcher,
        weight_dict=weight_dict, focal_alpha=cfg['focal_alpha'], losses=losses)
    return model, criterion
