import paddle
import os
import tqdm
import numpy as np
from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from utils import misc

from scipy import io


class Trainer(object):

    def __init__(self, cfg, model, optimizer, train_loader, test_loader,
        lr_scheduler, warmup_lr_scheduler, logger, loss, model_name):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.best_result = 0
        self.best_epoch = 0
        self.detr_loss = loss
        self.model_name = model_name
        self.output_dir = os.path.join('./' + cfg['save_path'], model_name)
        self.tester = None
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model, optimizer=None, filename=cfg[
                'pretrain_model'], logger=self.logger
                )
        if cfg.get('resume_model', None):
            resume_model_path = os.path.join(self.output_dir, 'checkpoint.pdparams')
            assert os.path.exists(resume_model_path)
            self.epoch, self.best_result, self.best_epoch = load_checkpoint(
                model=self.model, optimizer=self.optimizer,
                filename=resume_model_path,
                logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1
            self.logger.info(
                'Loading Checkpoint... Best Result:{}, Best Epoch:{}'.
                format(self.best_result, self.best_epoch))

    def train(self):
        start_epoch = self.epoch
        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']),
            dynamic_ncols=True, leave=True, desc='epochs')
        best_result = self.best_result
        best_epoch = self.best_epoch
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            np.random.seed(np.random.get_state()[1][0] + epoch)
            self.train_one_epoch(epoch)
            self.epoch += 1
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()
            if self.epoch % self.cfg['save_frequency'] == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                if self.cfg['save_all']:
                    ckpt_name = os.path.join(self.output_dir, 
                        'checkpoint_epoch_%d' % self.epoch)
                else:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint')
                save_checkpoint(get_checkpoint_state(self.model, self.
                    optimizer, self.epoch, best_result, best_epoch), ckpt_name)
                if self.tester is not None:
                    self.logger.info('Test Epoch {}'.format(self.epoch))
                    self.tester.inference()
                    cur_result = self.tester.evaluate()
                    if cur_result > best_result:
                        best_result = cur_result
                        best_epoch = self.epoch
                        ckpt_name = os.path.join(self.output_dir,
                            'checkpoint_best')
                        save_checkpoint(get_checkpoint_state(self.model,
                            self.optimizer, self.epoch, best_result,
                            best_epoch), ckpt_name)
                    self.logger.info('Best Result:{}, epoch:{}'.format(
                        best_result, best_epoch))
            progress_bar.update()
        self.logger.info('Best Result:{}, epoch:{}'.format(best_result,
            best_epoch))
        return None

    def train_one_epoch(self, epoch):
        paddle.set_grad_enabled(mode=True)
        self.model.train()
        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=self.epoch + 1 == self.cfg['max_epoch'], desc='iters')
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.train_loader):
            img_sizes = targets['img_size']
            targets = self.prepare_targets(targets, inputs.shape[0])
            self.optimizer.clear_grad()
            outputs = self.model(inputs, calibs, targets, img_sizes)
            detr_losses_dict = self.detr_loss(outputs, targets)
            weight_dict = self.detr_loss.weight_dict
            detr_losses_dict_weighted = [(detr_losses_dict[k] * weight_dict
                [k]) for k in detr_losses_dict.keys() if k in weight_dict]
            detr_losses = sum(detr_losses_dict_weighted)
            detr_losses_dict = misc.reduce_dict(detr_losses_dict)
            detr_losses_dict_log = {}
            detr_losses_log = 0
            for k in detr_losses_dict.keys():
                if k in weight_dict:
                    detr_losses_dict_log[k] = (detr_losses_dict[k] *
                        weight_dict[k]).item()
                    detr_losses_log += detr_losses_dict_log[k]
            detr_losses_dict_log['loss_detr'] = detr_losses_log
            flags = [True] * 5
            if batch_idx % 30 == 0:
                print('----', batch_idx, '----')
                print('%s: %.2f, ' % ('loss_detr', detr_losses_dict_log[
                    'loss_detr']))
                for key, val in detr_losses_dict_log.items():
                    if key == 'loss_detr':
                        continue
                    if ('0' in key or '1' in key or '2' in key or '3' in
                        key or '4' in key or '5' in key):
                        if flags[int(key[-1])]:
                            print('')
                            flags[int(key[-1])] = False
                    print('%s: %.2f, ' % (key, val), end='')
                print('')
                print('')
            detr_losses.backward()
            self.optimizer.step()
            progress_bar.update()
        progress_bar.close()

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']
        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d',
            'heading_bin', 'heading_res', 'boxes_3d']
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
            targets_list.append(target_dict)
        return targets_list
