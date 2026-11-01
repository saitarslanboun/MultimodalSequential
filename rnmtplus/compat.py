''' Compatibility with PyTorch 1.1 to 2.x and the matching torchvision versions. '''

import inspect

import torch
from torchvision import models

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# The ImageNet weights that resnet50(pretrained=True) loads in torchvision 0.3, used for the experiments.
RESNET50_URL = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'


def torch_load(path, map_location=None):
    ''' torch.load for data.pt files and checkpoints, which contain an argparse.Namespace.

    Since PyTorch 2.6, torch.load only loads tensors unless weights_only=False is given. '''
    kwargs = {}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, map_location=map_location, **kwargs)


def set_tf32(enabled):
    ''' Whether convolutions, GRUs and matrix products may use TensorFloat-32 on GPUs that have it (Ampere and
    later): faster, but less precise than float32, which the experiments were run with. PyTorch allows it
    for cuDNN by default. '''
    backends = torch.backends
    cuda = getattr(backends, 'cuda', None)
    matmul = getattr(cuda, 'matmul', None)
    if matmul is not None and hasattr(matmul, 'fp32_precision'):  # PyTorch 2.9 and later
        precision = 'tf32' if enabled else 'ieee'
        matmul.fp32_precision = precision
        for name in ('conv', 'rnn'):
            if hasattr(getattr(backends.cudnn, name, None), 'fp32_precision'):
                getattr(backends.cudnn, name).fp32_precision = precision
    elif hasattr(backends.cudnn, 'allow_tf32'):  # PyTorch 1.7 to 2.8
        matmul.allow_tf32 = enabled
        backends.cudnn.allow_tf32 = enabled


def resnet50(pretrained):
    ''' ResNet-50, with the ImageNet weights of the experiments if pretrained. '''
    if hasattr(models, 'ResNet50_Weights'):  # torchvision 0.13 and later
        resnet = models.resnet50(weights=None)
    else:
        resnet = models.resnet50(pretrained=False)
    if pretrained:
        resnet.load_state_dict(load_state_dict_from_url(RESNET50_URL, progress=True))
    return resnet
