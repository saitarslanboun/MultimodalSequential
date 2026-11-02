''' Compatibility with PyTorch 1.1 to 2.x and the matching torchvision versions. '''

import contextlib
import inspect
import os
import random
import sys

import numpy as np
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


def seed_python_hash(seed, extra_args=()):
    ''' Python draws the seed of its string hashes (which set the iteration order of sets) when it starts,
    unless PYTHONHASHSEED is set. If it is not the seed, runs the script again with PYTHONHASHSEED=seed, and
    extra_args added to its arguments; call it before the script does any work. '''
    if os.environ.get('PYTHONHASHSEED') == str(seed):
        return
    if sys.flags.ignore_environment:  # python -E or -I: PYTHONHASHSEED would be ignored
        print('[Warning] PYTHONHASHSEED is ignored with python -E or -I: the string hashes are not seeded')
        return
    os.environ['PYTHONHASHSEED'] = str(seed)
    sys.stdout.flush()
    sys.stderr.flush()
    argv = list(getattr(sys, 'orig_argv', [sys.executable] + sys.argv)) + list(extra_args)  # orig_argv: 3.10+
    os.execv(sys.executable, argv)


def seed_everything(seed):
    ''' Seeds Python's random, NumPy and PyTorch (CPU and every GPU). '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # done by torch.manual_seed too; explicit


def set_deterministic(enabled):
    ''' Deterministic algorithms instead of the fastest ones: cuDNN's, and in PyTorch 1.8 and later those of every
    operation (an operation without one raises an error). Call it before the first CUDA operation: cuBLAS is
    only deterministic with CUBLAS_WORKSPACE_CONFIG set when it starts (CUDA 10.2 and later). '''
    torch.backends.cudnn.benchmark = not enabled
    torch.backends.cudnn.deterministic = enabled
    if enabled:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)


def amp_supported():
    ''' Whether this PyTorch has automatic mixed precision (torch.autocast, PyTorch 1.10 and later). '''
    return hasattr(torch, 'autocast')


def autocast(enabled):
    ''' Mixed precision on the GPU (float16, with float32 where PyTorch needs it) if enabled. '''
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast('cuda', dtype=torch.float16)


def grad_scaler(enabled):
    ''' The loss scaler of mixed-precision training (float16 gradients would underflow), or None. '''
    if not enabled:
        return None
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):  # PyTorch 2.3 and later
        return torch.amp.GradScaler('cuda')
    return torch.cuda.amp.GradScaler()


def resnet50(pretrained):
    ''' ResNet-50, with the ImageNet weights of the experiments if pretrained. '''
    if hasattr(models, 'ResNet50_Weights'):  # torchvision 0.13 and later
        resnet = models.resnet50(weights=None)
    else:
        resnet = models.resnet50(pretrained=False)
    if pretrained:
        resnet.load_state_dict(load_state_dict_from_url(RESNET50_URL, progress=True))
    return resnet
