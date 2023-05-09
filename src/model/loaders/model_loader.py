import os
import torch
from util.gpu_util import use_gpu, first_device, all_devices


_checkpoint_path = '../../../data/model/{}.torch'
_model = None
_optimizer = None
_checkpoint = {}


def load_model(model_class, model_params=None):
    """
    create and eventually load model
    :param model_name:
    :param model_class:
    :param model_params:
    :param model_name:
    :return:
    """
    model_name = model_params["model_name"]
    path = model_params["model_path"]

    model = model_class(**model_params)
    _load_model(model, model_name, path=path)

    # configure usage on GPU
    if use_gpu():
        model.to(first_device())
        model = torch.nn.DataParallel(model, device_ids=all_devices())

    return model


def _load_model(model, model_name, path=None, reload=False):
    """
    load checkpoint
    :param model:
    :param model_name:
    :return:
    """
    global _checkpoint
    if model_name not in _checkpoint or reload:
        _load_checkpoint(model_name, path=path)

    if 'model_state_dict' in _checkpoint[model_name]:
        model.load_state_dict(_checkpoint[model_name]['model_state_dict'])
    else:
        model.load_state_dict(_checkpoint[model_name])



def _load_checkpoint(model_name, path=None):

    global _checkpoint
    if not os.path.isfile(path):
        print('{} does not exist'.format(path))
        raise Exception('{} does not exist when loading checkpoint.'.format(path))
    print('Loading checkpoint from ' + path)
    _checkpoint[model_name] = torch.load(path)


# # Not used here
# def load_checkpoint(model, model_name='model', validation_id=None):
#     """
#     change state of the model
#     """
#     path = output_path(_checkpoint_path.format(model_name), validation_id=validation_id, have_validation=True)
#     _load_model(model.module if type(model) is torch.nn.DataParallel else model, model_name, path=path, reload=True)


def create_optimizer(parameters, optimizer_class, optim_params, model_name='model'):
    """
    create and eventually load optimizer
    :param model_name:
    :param parameters:
    :param optimizer_class:
    :param optim_params:
    :return:
    """
    opt = optimizer_class(parameters, **optim_params)
    _load_optimizer(opt, model_name)
    return opt


def _load_optimizer(optimizer, model_name):
    """
    load checkpoint
    :param optimizer:
    :return:
    """
    global _checkpoint
    if model_name not in _checkpoint:
        _load_checkpoint(model_name)

    if 'optimizer_state_dict' in _checkpoint[model_name]:
        optimizer.load_state_dict(_checkpoint[model_name]['optimizer_state_dict'])
