import paddle
import os


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()
    for key, val in model_state.items():
        model_state_cpu[key] = val
    return model_state_cpu


def get_checkpoint_state(model=None, optimizer=None, epoch=None,
    best_result=None, best_epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        model_state = model.state_dict()
    else:
        model_state = None
    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state':
        optim_state, 'best_result': best_result, 'best_epoch': best_epoch}


def save_checkpoint(state, filename):
    filename = '{}.pdparams'.format(filename)
    paddle.save(state, filename)


def load_checkpoint(model, optimizer, filename, logger=None):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = paddle.load(filename)
        epoch = checkpoint.get('epoch', -1)
        best_result = checkpoint.get('best_result', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            model.set_state_dict(state_dict=checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.set_state_dict(state_dict=checkpoint['optimizer_state'])
        logger.info('==> Done')
    else:
        raise FileNotFoundError
    return epoch, best_result, best_epoch
