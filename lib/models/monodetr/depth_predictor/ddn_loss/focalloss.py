import paddle
import warnings
from typing import Optional


def one_hot(labels: paddle.Tensor, num_classes: int, dtype: Optional[paddle.dtype]=None, eps: float=1e-06
    ) ->paddle.Tensor:
    """Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        dtype: the desired data type of returned tensor.
    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples:
         labels = paddle.LongTensor([[[0, 1], [2, 0]]])
         one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])
    """
    if not isinstance(labels, paddle.Tensor):
        raise TypeError(
            f'Input labels type is not a paddle.Tensor. Got {type(labels)}')
    if not labels.dtype == paddle.int64:
        raise ValueError(
            f'labels must be of the same dtype paddle.int64. Got: {labels.dtype}'
            )
    if num_classes < 1:
        raise ValueError(
            'The number of classes must be bigger than one. Got: {}'.format
            (num_classes))
    shape = labels.shape
    one_hot = paddle.zeros(shape=[shape[0], num_classes]+ shape[1:],
        dtype=dtype)
    return paddle.put_along_axis(one_hot, indices=labels.unsqueeze(axis=1), values=1.0, axis=1)  + eps


def focal_loss(input: paddle.Tensor, target: paddle.Tensor, alpha: float,
    gamma: float=2.0, reduction: str='none', eps: Optional[float]=None
    ) ->paddle.Tensor:
    """Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \\text{FL}(p_t) = -\\alpha_t (1 - p_t)^{\\gamma} \\, \\text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\\alpha \\in [0, 1]`.
        gamma: Focusing parameter :math:`\\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stabiliy. This is no longer used.
    Return:
        the computed loss.
    Example:
         N = 5  # num_classes
         input = paddle.randn(1, N, 3, 5, requires_grad=True)
         target = paddle.empty(1, 3, 5, dtype=paddle.long).random_(N)
         output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
         output.backward()
    """
    if eps is not None:
        warnings.warn(
            '`focal_loss` has been reworked for improved numerical stability and the `eps` argument is no longer necessary'
            , DeprecationWarning, stacklevel=2)
    if not isinstance(input, paddle.Tensor):
        raise TypeError(f'Input type is not a paddle.Tensor. Got {type(input)}')
    if not len(input.shape) >= 2:
        raise ValueError(
            f'Invalid input shape, we expect BxCx*. Got: {input.shape}')
    if input.shape[0] != target.shape[0]:
        raise ValueError(
            f'Expected input batch_size ({input.shape[0]}) to match target batch_size ({target.shape[0]}).'
            )
    n = input.shape[0]
    out_size = [n,] + input.shape[2:]
    if target.shape[1:] != input.shape[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.shape}'
            )
    input_soft = paddle.nn.functional.softmax(x=input, axis=1)
    log_input_soft = paddle.nn.functional.log_softmax(x=
        input, axis=1)
    target_one_hot = one_hot(target, num_classes=input.shape
        [1], dtype=input.dtype)
    weight = paddle.pow(x=-input_soft + 1.0, y=gamma)
    focal = -alpha * weight * log_input_soft
    # loss_tmp = paddle.einsum('bc...,bc...->b...', (target_one_hot, focal))
    loss_tmp = paddle.sum(target_one_hot * focal, axis=1)
    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = paddle.mean(x=loss_tmp)
    elif reduction == 'sum':
        loss = paddle.sum(x=loss_tmp)
    else:
        raise NotImplementedError(f'Invalid reduction mode: {reduction}')
    return loss


class FocalLoss(paddle.nn.Layer):
    """Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \\text{FL}(p_t) = -\\alpha_t (1 - p_t)^{\\gamma} \\, \\text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        alpha: Weighting factor :math:`\\alpha \\in [0, 1]`.
        gamma: Focusing parameter :math:`\\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stability. This is no longer
          used.
    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Example:
         N = 5  # num_classes
         kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
         criterion = FocalLoss(**kwargs)
         input = paddle.randn(1, N, 3, 5, requires_grad=True)
         target = paddle.empty(1, 3, 5, dtype=paddle.long).random_(N)
         output = criterion(input, target)
         output.backward()
    """

    def __init__(self, alpha: float, gamma: float=2.0, reduction: str=
        'none', eps: Optional[float]=None) ->None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: Optional[float] = eps

    def forward(self, input: paddle.Tensor, target: paddle.Tensor
        ) ->paddle.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.
            reduction, self.eps)


def binary_focal_loss_with_logits(input: paddle.Tensor, target: paddle.
    Tensor, alpha: float=0.25, gamma: float=2.0, reduction: str='none', eps:
    Optional[float]=None) ->paddle.Tensor:
    """Function that computes Binary Focal loss.
    .. math::
        \\text{FL}(p_t) = -\\alpha_t (1 - p_t)^{\\gamma} \\, \\text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input: input data tensor of arbitrary shape.
        target: the target tensor with shape matching input.
        alpha: Weighting factor for the rare class :math:`\\alpha \\in [0, 1]`.
        gamma: Focusing parameter :math:`\\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar for numerically stability when dividing. This is no longer used.
    Returns:
        the computed loss.
    Examples:
         kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
         logits = paddle.tensor([[[6.325]],[[5.26]],[[87.49]]])
         labels = paddle.tensor([[[1.]],[[1.]],[[0.]]])
         binary_focal_loss_with_logits(logits, labels, **kwargs)
        tensor(21.8725)
    """
    if eps is not None:
        warnings.warn(
            '`binary_focal_loss_with_logits` has been reworked for improved numerical stability and the `eps` argument is no longer necessary'
            , DeprecationWarning, stacklevel=2)
    if not isinstance(input, paddle.Tensor):
        raise TypeError(f'Input type is not a paddle.Tensor. Got {type(input)}')
    if not len(input.shape) >= 2:
        raise ValueError(
            f'Invalid input shape, we expect BxCx*. Got: {input.shape}')
    if input.shape[0] != target.shape[0]:
        raise ValueError(
            f'Expected input batch_size ({input.shape[0]}) to match target batch_size ({target.shape[0]}).'
            )
    probs_pos = paddle.nn.functional.sigmoid(x=input)
    probs_neg = paddle.nn.functional.sigmoid(x=-input)
    loss_tmp = -alpha * paddle.pow(x=probs_neg, y=gamma
        ) * target * paddle.nn.functional.logsigmoid(x=input) - (1 - alpha
        ) * paddle.pow(x=probs_pos, y=gamma) * (1.0 - target
        ) * paddle.nn.functional.logsigmoid(x=-input)
    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = paddle.mean(x=loss_tmp)
    elif reduction == 'sum':
        loss = paddle.sum(x=loss_tmp)
    else:
        raise NotImplementedError(f'Invalid reduction mode: {reduction}')
    return loss


class BinaryFocalLossWithLogits(paddle.nn.Layer):
    """Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \\text{FL}(p_t) = -\\alpha_t (1 - p_t)^{\\gamma} \\, \\text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        alpha): Weighting factor for the rare class :math:`\\alpha \\in [0, 1]`.
        gamma: Focusing parameter :math:`\\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
    Shape:
        - Input: :math:`(N, *)`.
        - Target: :math:`(N, *)`.
    Examples:
         kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
         loss = BinaryFocalLossWithLogits(**kwargs)
         input = paddle.randn(1, 3, 5, requires_grad=True)
         target = paddle.empty(1, 3, 5, dtype=paddle.long).random_(2)
         output = loss(input, target)
         output.backward()
    """

    def __init__(self, alpha: float, gamma: float=2.0, reduction: str='none'
        ) ->None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction

    def forward(self, input: paddle.Tensor, target: paddle.Tensor
        ) ->paddle.Tensor:
        return binary_focal_loss_with_logits(input, target, self.alpha,
            self.gamma, self.reduction)
