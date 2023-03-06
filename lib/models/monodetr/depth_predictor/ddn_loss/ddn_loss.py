import paddle
import math
from .balancer import Balancer
from .focalloss import FocalLoss


class DDNLoss(paddle.nn.Layer):

    def __init__(self, alpha=0.25, gamma=2.0, fg_weight=13, bg_weight=1,
        downsample_factor=1):
        """
        Initializes DDNLoss module
        Args:
            weight [float]: Loss function weight
            alpha [float]: Alpha value for Focal Loss
            gamma [float]: Gamma value for Focal Loss
            disc_cfg [dict]: Depth discretiziation configuration
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Depth map downsample factor
        """
        super().__init__()
        self.balancer = Balancer(downsample_factor=downsample_factor,
            fg_weight=fg_weight, bg_weight=bg_weight)
        self.alpha = alpha
        self.gamma = gamma
        self.loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma,
            reduction='none')

    def build_target_depth_from_3dcenter(self, depth_logits, gt_boxes2d,
        gt_center_depth, num_gt_per_img):
        B, _, H, W = depth_logits.shape
        depth_maps = paddle.zeros(shape=(B, H, W), dtype=depth_logits.dtype)
        gt_boxes2d[:, :2] = paddle.floor(x=gt_boxes2d[:, :2])
        gt_boxes2d[:, 2:] = paddle.ceil(x=gt_boxes2d[:, 2:])
        gt_boxes2d = gt_boxes2d.astype(dtype='int64')
        gt_boxes2d = paddle.split(gt_boxes2d, num_gt_per_img, axis=0)
        gt_center_depth = paddle.split(gt_center_depth, num_gt_per_img, axis=0)
        B = len(gt_boxes2d)
        for b in range(B):
            center_depth_per_batch = gt_center_depth[b]
            sorted_idx = paddle.argsort(x=
                center_depth_per_batch, axis=0, descending=True)
            center_depth_per_batch = paddle.sort(x=
                center_depth_per_batch, axis=0, descending=True)
            if gt_boxes2d[b].shape[0] != 0:
                gt_boxes_per_batch = gt_boxes2d[b][sorted_idx]
                if sorted_idx.shape[0] == 1:
                    gt_boxes_per_batch = gt_boxes_per_batch.unsqueeze(0)
                for n in range(gt_boxes_per_batch.shape[0]):
                    u1, v1, u2, v2 = gt_boxes_per_batch[n]
                    if u1>=0 and v1>=0:
                        depth_maps[b, v1:v2, u1:u2] = center_depth_per_batch[n]
        return depth_maps

    def bin_depths(self, depth_map, mode='LID', depth_min=0.001, depth_max=
        60, num_bins=80, target=False):
        """
        Converts depth map into bin indices
        Args:
            depth_map [torch.Tensor(H, W)]: Depth Map
            mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
                UD: Uniform discretiziation
                LID: Linear increasing discretiziation
                SID: Spacing increasing discretiziation
            depth_min [float]: Minimum depth value
            depth_max [float]: Maximum depth value
            num_bins [int]: Number of depth bins
            target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
        Returns:
            indices [torch.Tensor(H, W)]: Depth bin indices
        """
        if mode == 'UD':
            bin_size = (depth_max - depth_min) / num_bins
            indices = (depth_map - depth_min) / bin_size
        elif mode == 'LID':
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins)
                )
            indices = -0.5 + 0.5 * paddle.sqrt(x=1 + 8 * (depth_map -
                depth_min) / bin_size)
        elif mode == 'SID':
            indices = num_bins * (paddle.log(x=1 + depth_map) - math.log(1 +
                depth_min)) / (math.log(1 + depth_max) - math.log(1 +
                depth_min))
        else:
            raise NotImplementedError
        if target:
            mask = (indices < 0) | (indices > num_bins) | ~paddle.isfinite(x
                =indices)
            indices[mask] = num_bins
            indices = indices.cast('int64')
        return indices

    def forward(self, depth_logits, gt_boxes2d, num_gt_per_img, gt_center_depth
        ):
        """
        Gets depth_map loss
        Args:
            depth_logits: torch.Tensor(B, D+1, H, W)]: Predicted depth logits
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
            num_gt_per_img:
            gt_center_depth:
        Returns:
            loss [torch.Tensor(1)]: Depth classification network loss
        """
        depth_maps = self.build_target_depth_from_3dcenter(depth_logits,
            gt_boxes2d, gt_center_depth, num_gt_per_img)
        
        depth_target = self.bin_depths(depth_maps, target=True)
        loss = self.loss_func(depth_logits, depth_target)
        loss = self.balancer(loss=loss, gt_boxes2d=gt_boxes2d,
            num_gt_per_img=num_gt_per_img)
        return loss
