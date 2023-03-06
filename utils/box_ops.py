import paddle

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(axis=-1)
    b = [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h]
    return paddle.stack(x=b, axis=-1)


def box_cxcylrtb_to_xyxy(x):
    x_c, y_c, l, r, t, b = x.unbind(axis=-1)
    bb = [x_c - l, y_c - t, x_c + r, y_c + b]
    return paddle.stack(x=bb, axis=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(axis=-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    return paddle.stack(x=b, axis=-1)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = paddle.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = paddle.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)
    lt = paddle.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = paddle.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clip(min=0)
    area = wh[:, :, (0)] * wh[:, :, (1)]
    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return paddle.zeros(shape=(0, 4))
    h, w = masks.shape[-2:]
    y = paddle.arange(start=0, end=h, dtype='float32')
    x = paddle.arange(start=0, end=w, dtype='float32')
    y, x = paddle.meshgrid(y, x)
    x_mask = masks * x.unsqueeze(axis=0)
    x_max = x_mask.flatten(start_axis=1).max(axis=-1)[0]
    x_min = paddle.where(~masks.cast('bool'), 
        paddle.full(x_mask.shape, 100000000.0, x_mask.dtype), 
        x_mask).flatten(start_axis=1).logsumexp(axis=-1)[0]
    y_mask = masks * y.unsqueeze(axis=0)
    y_max = y_mask.flatten(start_axis=1).max(axis=-1)
    y_min = paddle.where(~masks.cast('bool'), 
        paddle.full(y_mask.shape, 100000000.0, y_mask.dtype), 
        y_mask).flatten(start_axis=1).logsumexp(axis=-1)[0]
    return paddle.stack(x=[x_min, y_min, x_max, y_max], axis=1)
