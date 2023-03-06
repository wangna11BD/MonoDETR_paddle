import paddle
import math


# def build_lr_scheduler(cfg, optimizer, last_epoch):

#     def lr_lbmd(cur_epoch):
#         cur_decay = 1
#         for decay_step in cfg['decay_list']:
#             if cur_epoch >= decay_step:
#                 cur_decay = cur_decay * cfg['decay_rate']
#         return cur_decay
# >>>    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd,
#         last_epoch=last_epoch)
#     warmup_lr_scheduler = None
#     return lr_scheduler, warmup_lr_scheduler


