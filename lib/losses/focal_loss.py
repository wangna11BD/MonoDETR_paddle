import paddle


def focal_loss(input, target, alpha=0.25, gamma=2.0):
    """
    Args:
        input:  prediction, 'batch x c x h x w'
        target:  ground truth, 'batch x c x h x w'
        alpha: hyper param, default in 0.25
        gamma: hyper param, default in 2.0
    Reference: Focal Loss for Dense Object Detection, ICCV'17
    """
    pos_inds = target.equal(y=1).cast('float32')
    neg_inds = target.less_than(y=1).cast('float32')
    loss = 0
    pos_loss = paddle.log(x=input) * paddle.pow(x=1 - input, y=gamma
        ) * pos_inds * alpha
    neg_loss = paddle.log(x=1 - input) * paddle.pow(x=input, y=gamma
        ) * neg_inds * (1 - alpha)
    num_pos = pos_inds.cast('float32').sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss.mean()


def focal_loss_cornernet(input, target, gamma=2.0):
    """
    Args:
        input:  prediction, 'batch x c x h x w'
        target:  ground truth, 'batch x c x h x w'
        gamma: hyper param, default in 2.0
    Reference: Cornernet: Detecting Objects as Paired Keypoints, ECCV'18
    """
    pos_inds = target.equal(y=1).cast('float32')
    neg_inds = target.less_than(y=1).cast('float32')
    neg_weights = paddle.pow(x=1 - target, y=4)
    loss = 0
    pos_loss = paddle.log(x=input) * paddle.pow(x=1 - input, y=gamma
        ) * pos_inds
    neg_loss = paddle.log(x=1 - input) * paddle.pow(x=input, y=gamma
        ) * neg_inds * neg_weights
    num_pos = pos_inds.cast('float32').sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss.mean()


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float=0.25, gamma:
    float=2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = paddle.nn.functional.sigmoid(inputs)
    ce_loss = paddle.nn.functional.binary_cross_entropy_with_logits(inputs,
        targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * (1 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(axis=1).sum() / num_boxes
