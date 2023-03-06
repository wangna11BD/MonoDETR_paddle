import paddle
import math


def build_optimizer(cfg_optimizer, lr, model):
    weights, biases = [], []
    for name, param in model.named_parameters():
        if 'bias' in name:
            biases += [param]
        else:
            weights += [param]
    parameters = [{'params': biases, 'weight_decay': 0}, {'params': weights,
        'weight_decay': cfg_optimizer['weight_decay']}]
    if cfg_optimizer['type'] == 'adamw':
        optimizer = paddle.optimizer.AdamW(parameters=parameters, learning_rate=lr,
                        weight_decay=0.0)
    else:
        raise NotImplementedError('%s optimizer is not supported' %
            cfg_optimizer['type'])
    return optimizer


# >>>class AdamW(torch.optim.optimizer.Optimizer):
#     """Implements Adam algorithm.
#     It has been proposed in `Adam: A Method for Stochastic Optimization`_.
#     Arguments:
#         params (iterable): iterable of parameters to optimize or dicts defining
#             parameter groups
#         lr (float, optional): learning rate (default: 1e-3)
#         betas (Tuple[float, float], optional): coefficients used for computing
#             running averages of gradient and its square (default: (0.9, 0.999))
#         eps (float, optional): term added to the denominator to improve
#             numerical stability (default: 1e-8)
#         weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
#         amsgrad (boolean, optional): whether to use the AMSGrad variant of this
#             algorithm from the paper `On the Convergence of Adam and Beyond`_
#     .. _Adam\\: A Method for Stochastic Optimization:
#         https://arxiv.org/abs/1412.6980
#     .. _On the Convergence of Adam and Beyond:
#         https://openreview.net/forum?id=ryQu7f-RZ
#     """

#     def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08,
#         weight_decay=0, amsgrad=False):
#         if not 0.0 <= lr:
#             raise ValueError('Invalid learning rate: {}'.format(lr))
#         if not 0.0 <= eps:
#             raise ValueError('Invalid epsilon value: {}'.format(eps))
#         if not 0.0 <= betas[0] < 1.0:
#             raise ValueError('Invalid beta parameter at index 0: {}'.format
#                 (betas[0]))
#         if not 0.0 <= betas[1] < 1.0:
#             raise ValueError('Invalid beta parameter at index 1: {}'.format
#                 (betas[1]))
#         defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=
#             weight_decay, amsgrad=amsgrad)
#         super(AdamW, self).__init__(params, defaults)

#     def __setstate__(self, state):
#         super(AdamW, self).__setstate__(state)
#         for group in self.param_groups:
#             group.setdefault('amsgrad', False)

#     def step(self, closure=None):
#         """Performs a single optimization step.
#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = None
#         if closure is not None:
#             loss = closure()
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad.data
#                 amsgrad = group['amsgrad']
#                 state = self.state[p]
#                 if len(state) == 0:
#                     state['step'] = 0
#                     state['exp_avg'] = paddle.zeros_like(x=p.data)
#                     state['exp_avg_sq'] = paddle.zeros_like(x=p.data)
#                     if amsgrad:
#                         state['max_exp_avg_sq'] = paddle.zeros_like(x=p.data)
#                 exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#                 if amsgrad:
#                     max_exp_avg_sq = state['max_exp_avg_sq']
#                 beta1, beta2 = group['betas']
#                 state['step'] += 1
#                 """Class Method: *.add_, not convert, please check which one of torch.Tensor.*/Optimizer.*/nn.Module.* it is, and convert manually"""
# >>>                exp_avg.scale_(scale=beta1).add_(1 - beta1, grad)
#                 """Class Method: *.addcmul_, not convert, please check which one of torch.Tensor.*/Optimizer.*/nn.Module.* it is, and convert manually"""
# >>>                exp_avg_sq.scale_(scale=beta2).addcmul_(1 - beta2, grad, grad)
#                 if amsgrad:
#                     max_exp_avg_sq = paddle.maximum(x=max_exp_avg_sq, y=
#                         exp_avg_sq)
#                     max_exp_avg_sq
#                     """Class Method: *.add_, not convert, please check which one of torch.Tensor.*/Optimizer.*/nn.Module.* it is, and convert manually"""
# >>>                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
#                 else:
#                     """Class Method: *.add_, not convert, please check which one of torch.Tensor.*/Optimizer.*/nn.Module.* it is, and convert manually"""
# >>>                    denom = exp_avg_sq.sqrt().add_(group['eps'])
#                 bias_correction1 = 1 - beta1 ** state['step']
#                 bias_correction2 = 1 - beta2 ** state['step']
#                 step_size = group['lr'] * math.sqrt(bias_correction2
#                     ) / bias_correction1
#                 """Class Method: *.addcdiv_, not convert, please check which one of torch.Tensor.*/Optimizer.*/nn.Module.* it is, and convert manually"""
# >>>                p.data.add_(-step_size, paddle.multiply(x=p.data, y=group[
#                     'weight_decay']).addcdiv_(1, exp_avg, denom))
#         return loss
