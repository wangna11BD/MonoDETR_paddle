import paddle
import numpy as np
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, workers=0):
    if cfg['type'] == 'KITTI':
        train_set = KITTI_Dataset(split=cfg['train_split'], cfg=cfg)
        test_set = KITTI_Dataset(split=cfg['test_split'], cfg=cfg)
    else:
        raise NotImplementedError('%s dataset is not supported' % cfg['type'])
    train_loader = paddle.io.DataLoader(dataset=train_set,
        batch_size=cfg['batch_size'], num_workers=workers, worker_init_fn=
        my_worker_init_fn, shuffle=True, drop_last=False)
    test_loader = paddle.io.DataLoader(dataset=test_set, batch_size=
        cfg['batch_size_test'], num_workers=workers, worker_init_fn=
        my_worker_init_fn, shuffle=False, drop_last=False)
    return train_loader, test_loader
