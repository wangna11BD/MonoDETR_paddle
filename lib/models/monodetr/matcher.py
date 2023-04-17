import paddle
import scipy
from scipy.optimize import linear_sum_assignment
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_xyxy_to_cxcywh, box_cxcylrtb_to_xyxy
from paddle3d.models.heads.dense_heads.match_costs.match_cost import pairwise_dist

class HungarianMatcher(paddle.nn.Layer):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float=1, cost_3dcenter: float=1,
        cost_bbox: float=1, cost_giou: float=1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_3dcenter = cost_3dcenter
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'

    @paddle.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs['pred_boxes'].shape[:2]
        out_prob = paddle.nn.functional.sigmoid(outputs['pred_logits'].flatten(start_axis=0, stop_axis=1))
        tgt_ids = paddle.concat(x=[v['labels'] for v in targets]).astype(dtype ='int64')
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * out_prob ** gamma * -(1 - out_prob + 1e-08).log()
        pos_cost_class = alpha * (1 - out_prob) ** gamma * -(out_prob + 1e-08).log()
        # cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        cost_class = paddle.index_select(pos_cost_class, tgt_ids, axis=1) - paddle.index_select(neg_cost_class, tgt_ids, axis=1)
        out_3dcenter = outputs['pred_boxes'][:, :, 0:2].flatten(start_axis=
            0, stop_axis=1)
        tgt_3dcenter = paddle.concat(x=[v['boxes_3d'][:, 0:2] for v in targets])

        cost_3dcenter = pairwise_dist(out_3dcenter, tgt_3dcenter)

        out_2dbbox = outputs['pred_boxes'][:, :, 2:6].flatten(start_axis=0,
            stop_axis=1)
        tgt_2dbbox = paddle.concat(x=[v['boxes_3d'][:, 2:6] for v in targets])
        cost_bbox = pairwise_dist(out_2dbbox, tgt_2dbbox)

        out_bbox = outputs['pred_boxes'].flatten(start_axis=0, stop_axis=1)
        tgt_bbox = paddle.concat(x=[v['boxes_3d'] for v in targets])
        cost_giou = -generalized_box_iou(box_cxcylrtb_to_xyxy(out_bbox),
            box_cxcylrtb_to_xyxy(tgt_bbox))
        C = (self.cost_bbox * cost_bbox + self.cost_3dcenter *
            cost_3dcenter + self.cost_class * cost_class + self.cost_giou *
            cost_giou)
        C = C.reshape([bs, num_queries, -1])
        sizes = [len(v['boxes']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(paddle.split(C, sizes, -1))]
        return [(paddle.to_tensor(data=i, dtype='int64'), paddle.to_tensor(
            data=j, dtype='int64')) for i, j in indices]


def build_matcher(cfg):
    return HungarianMatcher(cost_class=cfg['set_cost_class'], cost_bbox=cfg
        ['set_cost_bbox'], cost_3dcenter=cfg['set_cost_3dcenter'],
        cost_giou=cfg['set_cost_giou'])
