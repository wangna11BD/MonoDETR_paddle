import paddle
import numpy as np
from lib.datasets.utils import class2angle
from utils import box_ops


def decode_detections(dets, info, calibs, cls_mean_size, threshold):
    """
    NOTE: THIS IS A NUMPY FUNCTION
    input: dets, numpy array, shape in [batch x max_dets x dim]
    input: img_info, dict, necessary information of input images
    input: calibs, corresponding calibs for the input batch
    output:
    """
    results = {}
    for i in range(dets.shape[0]):
        preds = []
        for j in range(dets.shape[1]):
            cls_id = int(dets[i, j, 0])
            score = dets[i, j, 1]
            if score < threshold:
                continue
            x = dets[i, j, 2] * info['img_size'][i][0].numpy()[0]
            y = dets[i, j, 3] * info['img_size'][i][1].numpy()[0]
            w = dets[i, j, 4] * info['img_size'][i][0].numpy()[0]
            h = dets[i, j, 5] * info['img_size'][i][1].numpy()[0]
            bbox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
            depth = dets[i, j, 6]
            dimensions = dets[(i), (j), 31:34]
            dimensions += cls_mean_size[int(cls_id)]
            x3d = dets[i, j, 34] * info['img_size'][i][0].numpy()[0]
            y3d = dets[i, j, 35] * info['img_size'][i][1].numpy()[0]
            locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape([-1])
            locations[1] += dimensions[0] / 2
            alpha = get_heading_angle(dets[(i), (j), 7:31])
            ry = calibs[i].alpha2ry(alpha, x)
            score = score * dets[i, j, -1]
            preds.append([cls_id, alpha] + bbox + dimensions.tolist() +
                locations.tolist() + [ry, score])
        results[info['img_id'][i]] = preds
    return results


def extract_dets_from_outputs(outputs, K=50, topk=50):
    out_logits = outputs['pred_logits']
    out_bbox = outputs['pred_boxes']
    prob = paddle.nn.functional.sigmoid(out_logits)
    topk_values, topk_indexes = paddle.topk(x=prob.reshape([out_logits.shape[0],
        -1]), k=topk, axis=1)
    scores = topk_values
    topk_boxes = (topk_indexes // out_logits.shape[2]).unsqueeze(axis=-1)
    labels = topk_indexes % out_logits.shape[2]
    heading = outputs['pred_angle']
    size_3d = outputs['pred_3d_dim']
    depth = outputs['pred_depth'][:, :, 0:1]
    sigma = outputs['pred_depth'][:, :, 1:2]
    sigma = paddle.exp(x=-sigma)
    boxes = paddle.take_along_axis(arr=out_bbox, axis=1, indices=topk_boxes
        .tile(repeat_times=[1, 1, 6]))
    xs3d = boxes[:, :, 0:1]
    ys3d = boxes[:, :, 1:2]
    heading = paddle.take_along_axis(arr=heading, axis=1, indices=
        topk_boxes.tile(repeat_times=[1, 1, 24]))
    depth = paddle.take_along_axis(arr=depth, axis=1, indices=topk_boxes)
    sigma = paddle.take_along_axis(arr=sigma, axis=1, indices=topk_boxes)
    size_3d = paddle.take_along_axis(arr=size_3d, axis=1, indices=
        topk_boxes.tile(repeat_times=[1, 1, 3]))
    corner_2d = box_ops.box_cxcylrtb_to_xyxy(boxes)
    xywh_2d = box_ops.box_xyxy_to_cxcywh(corner_2d)
    size_2d = xywh_2d[:, :, 2:4]
    xs2d = xywh_2d[:, :, 0:1]
    ys2d = xywh_2d[:, :, 1:2]
    batch = out_logits.shape[0]
    labels = labels.reshape([batch, -1, 1])
    scores = scores.reshape([batch, -1, 1])
    xs2d = xs2d.reshape([batch, -1, 1])
    ys2d = ys2d.reshape([batch, -1, 1])
    xs3d = xs3d.reshape([batch, -1, 1])
    ys3d = ys3d.reshape([batch, -1, 1])
    detections = paddle.concat(x=[labels.cast('float32'), scores, xs2d, ys2d, size_2d,
        depth, heading, size_3d, xs3d, ys3d, sigma], axis=2)
    return detections


def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = paddle.nn.functional.max_pool2d(heatmap, (kernel, kernel),
        stride=1, padding=padding)
    keep = (heatmapmax == heatmap).cast('float32')
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.shape
    topk_scores, topk_inds = paddle.topk(x=heatmap.reshape([batch, cat, -1]), k=K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).cast('int64').cast('float32')
    topk_xs = (topk_inds % width).cast('int64').cast('float32')
    topk_score, topk_ind = paddle.topk(x=topk_scores.reshape([batch, -1]), k=K)
    topk_cls_ids = (topk_ind / K).cast('int64')
    topk_inds = _gather_feat(topk_inds.reshape([batch, -1, 1]), topk_ind).reshape([batch
        , K])
    topk_ys = _gather_feat(topk_ys.reshape([batch, -1, 1]), topk_ind).reshape([batch, K])
    topk_xs = _gather_feat(topk_xs.reshape([batch, -1, 1]), topk_ind).reshape([batch, K])
    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    """
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    """
    dim = feat.shape[2]
    ind = ind.unsqueeze(axis=2).expand([ind.shape[0], ind.shape[1], dim])
    feat = feat.take_along_axis(axis=1, index=ind)
    if mask is not None:
        mask = mask.unsqueeze(axis=2).expand_as(y=feat)
        feat = feat[mask]
        feat = feat.reshape([-1, dim])
    return feat


def _transpose_and_gather_feat(feat, ind):
    """
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    """
    feat = feat.transpose(perm=[0, 2, 3, 1])
    feat = feat.reshape([feat.shape[0], -1, feat.shape[3]])
    feat = _gather_feat(feat, ind)
    return feat


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)
