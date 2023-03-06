import paddle
import numpy as np


def laplacian_aleatoric_uncertainty_loss(input, target, log_variance,
    reduction='mean'):
    """
    References:
        MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships, CVPR'20
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    """
    assert reduction in ['mean', 'sum']
    loss = 1.4142 * paddle.exp(x=-log_variance) * paddle.abs(x=input - target
        ) + log_variance
    return loss.mean() if reduction == 'mean' else loss.sum()


def gaussian_aleatoric_uncertainty_loss(input, target, log_variance,
    reduction='mean'):
    """
    References:
        What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?, Neuips'17
        Geometry and Uncertainty in Deep Learning for Computer Vision, University of Cambridge
    """
    assert reduction in ['mean', 'sum']
    loss = 0.5 * paddle.exp(x=-log_variance) * paddle.abs(x=input - target
        ) ** 2 + 0.5 * log_variance
    return loss.mean() if reduction == 'mean' else loss.sum()


if __name__ == '__main__':
    pass
