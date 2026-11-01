''' Batches of images, titles and labels for training and prediction.

The batches are those of the original code: consecutive slices of the data, which is shuffled with Python's
random.shuffle when the loader is created and after every epoch; titles and token sequences are padded to the
longest in the batch. Images are read and transformed in worker processes (num_workers > 0), so that loading
overlaps with the computation on the GPU.

With a seed (train.py -kaggle_dir, translate.py), the order of each epoch is a permutation drawn with
numpy.random.RandomState([seed, epoch]), the worker processes are seeded from the seed, the epoch and their
number, and the random crop and flip of an image are drawn with a random.Random seeded with the seed, the
epoch and the index of the product (SeededRandomTransform, PredictionSeededTransform). The batches of every
epoch are then the same in all runs with that seed, whatever the number of worker processes.
'''

import functools
import math
import os
import random

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

import rnmtplus.Constants as Constants

# The target formats: token sequences (sequential), class indices (multiclass), one index per attribute.
TGT_FORMATS = ('sequence', 'chain', 'attributes')
CROPS = {'random_crop': 'RandomCrop', 'random_resized_crop': 'RandomResizedCrop', 'center_crop': 'CenterCrop'}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # of ImageNet, for the ResNet
RESIZE_RATIO = 256 / 224.  # center_crop: resize the shorter side to 256 for a crop of 224, as for ImageNet
ZERO_IMAGE_SIZE = 224  # the size of the zero image that replaces a missing or unreadable image


def image_transform(crop, crop_size=224):
    ''' The image preprocessing.
    random_crop, random_resized_crop: the random crop of the experiment (RandomCrop or RandomResizedCrop) and a
    random horizontal flip, which the paper's runs used in training, validation and prediction alike.
    center_crop: resize the shorter side to 256/224 of the crop size and crop the center; no randomness. '''
    if crop == 'center_crop':
        steps = [transforms.Resize(int(round(crop_size * RESIZE_RATIO))), transforms.CenterCrop(crop_size)]
    else:
        steps = [getattr(transforms, CROPS[crop])(crop_size), transforms.RandomHorizontalFlip()]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def _randint(rng, a, b):
    ''' A random integer from a to b inclusive, from rng.random() alone (its numbers are the same in every
    Python version). '''
    return a + min(int(rng.random() * (b - a + 1)), b - a)


def _resized_crop_params(img, rng, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
    ''' The crop of torchvision's RandomResizedCrop.get_params: top, left, height, width. '''
    width, height = img.size
    area = height * width
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * (scale[0] + (scale[1] - scale[0]) * rng.random())
        aspect_ratio = math.exp(log_ratio[0] + (log_ratio[1] - log_ratio[0]) * rng.random())
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < w <= width and 0 < h <= height:
            return _randint(rng, 0, height - h), _randint(rng, 0, width - w), h, w
    in_ratio = float(width) / float(height)  # fall back to a central crop
    if in_ratio < min(ratio):
        w, h = width, int(round(width / min(ratio)))
    elif in_ratio > max(ratio):
        w, h = int(round(height * max(ratio))), height
    else:
        w, h = width, height
    return (height - h) // 2, (width - w) // 2, h, w


class _SeededTransform(object):
    ''' An image transformation that is called with the image and a random.Random seeded for the image. '''


class SeededRandomTransform(_SeededTransform):
    ''' The random crop and flip of image_transform, drawn with the random.Random given with each image instead
    of the global random number generators. The crops are drawn as by torchvision's RandomCrop and
    RandomResizedCrop (area 8% to 100%, aspect ratio 3/4 to 4/3). '''

    def __init__(self, crop, crop_size=224):
        if crop not in ('random_crop', 'random_resized_crop'):
            raise ValueError('not a random crop: ' + crop)
        self.crop = crop
        self.crop_size = crop_size

    def __call__(self, img, rng):
        size = self.crop_size
        if self.crop == 'random_resized_crop':
            top, left, h, w = _resized_crop_params(img, rng)
            img = F.resized_crop(img, top, left, h, w, (size, size))
        else:
            width, height = img.size
            if width < size or height < size:
                raise ValueError('the image is smaller than the crop')
            top = _randint(rng, 0, height - size)
            img = F.crop(img, top, _randint(rng, 0, width - size), size, size)
        if rng.random() < 0.5:
            img = F.hflip(img)
        return F.normalize(F.to_tensor(img), MEAN, STD)


class PredictionSeededTransform(_SeededTransform):
    ''' A transformation of image_transform, run with Python's random, NumPy and PyTorch seeded for each image
    from the random.Random given with it: the random crops of the installed torchvision, seeded per product. It
    reseeds the global random number generators, so it is for prediction (translate.py), which draws no other
    random numbers. '''

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, rng):
        seed = rng.getrandbits(32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return self.transform(img)


def _seed_worker(seed, epoch, worker_id):
    ''' Seeds Python's random, NumPy and PyTorch in a worker process with a number drawn from the seed of the
    loader, the epoch and the number of the worker. '''
    worker_seed = random.Random('{} {} worker {}'.format(seed, epoch, worker_id)).getrandbits(32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def load_image(image_dir, img_inst, transform):
    ''' The transformed image; a zero image if the name is "None" or the file cannot be read or transformed. '''
    if type(img_inst) is list:  # names read with read_instances_from_file: [<BOS>, name, <EOS>]
        img_inst = img_inst[1]
    if img_inst == "None":
        return torch.zeros((3, ZERO_IMAGE_SIZE, ZERO_IMAGE_SIZE))
    try:
        return transform(Image.open(os.path.join(image_dir, img_inst)))
    except Exception:
        return torch.zeros((3, ZERO_IMAGE_SIZE, ZERO_IMAGE_SIZE))


def pad_to_longest(insts):
    ''' The token index lists as a LongTensor, padded with PAD to the longest. '''
    max_len = max(len(inst) for inst in insts)
    return torch.LongTensor([inst + [Constants.PAD] * (max_len - len(inst)) for inst in insts])


class _Instances(torch.utils.data.Dataset):

    def __init__(self, image_dir, transform, image_insts, src_insts, tgt_insts, seed=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_insts = image_insts
        self.src_insts = src_insts
        self.tgt_insts = tgt_insts
        self.seed = seed
        self.epoch = 0  # set by the DataLoader before each epoch

    def __len__(self):
        return len(self.src_insts if self.src_insts is not None else self.image_insts)

    def __getitem__(self, index):
        image = None
        if self.image_insts is not None:
            transform = self.transform
            if isinstance(transform, _SeededTransform):
                rng = random.Random('{} {} {}'.format(self.seed, self.epoch, index))
                transform = functools.partial(transform, rng=rng)
            image = load_image(self.image_dir, self.image_insts[index], transform)
        src = self.src_insts[index] if self.src_insts is not None else None
        tgt = self.tgt_insts[index] if self.tgt_insts is not None else None
        return index, image, src, tgt


class _Collate(object):

    def __init__(self, tgt_format):
        self.tgt_format = tgt_format

    def __call__(self, items):
        indices, images, srcs, tgts = zip(*items)
        image = torch.stack(images) if images[0] is not None else None
        src = pad_to_longest(srcs) if srcs[0] is not None else None
        if tgts[0] is None:
            tgt = None
        elif self.tgt_format == 'sequence':
            tgt = pad_to_longest(tgts)
        else:
            tgt = torch.LongTensor(list(tgts))
        return list(indices), image, src, tgt


class DataLoader(object):
    ''' Iterating over it gives the batches of an epoch as tuples (indices, image, src, tgt): the indices of the
    products in the data, the images (batch x 3 x 224 x 224), the padded titles (batch x length) and the
    targets (a padded batch x length tensor for 'sequence', batch for 'chain', batch x attributes for
    'attributes'); None for what the data does not have. '''

    def __init__(self, transform, image_dir, src_word2idx, tgt_word2idx, image_insts=None, src_insts=None,
                 tgt_insts=None, tgt_format='sequence', cuda=True, batch_size=64, shuffle=True, test=False,
                 num_workers=0, batches=None, seed=None):
        ''' image_insts, src_insts, tgt_insts: the instances of data.pt (preprocess.py); for the 'attributes'
        format, tgt_insts has one list per attribute. batches: lists of indices to use as the batches,
        instead of slices of batch_size (no shuffling). seed: shuffle and transform the images with this seed
        (see above) instead of the global random number generators. test: not used (kept from the original
        code). '''
        assert image_insts or src_insts
        assert tgt_format in TGT_FORMATS
        if isinstance(transform, _SeededTransform) and seed is None:
            raise ValueError('a seeded image transformation needs a seed')
        n_insts = len(src_insts) if src_insts else len(image_insts)
        if image_insts and src_insts:
            assert len(image_insts) == len(src_insts)
        if tgt_insts:
            if tgt_format == 'attributes':
                tgt_insts = list(map(list, zip(*tgt_insts)))
            assert len(tgt_insts) == n_insts
        self.cuda = cuda
        self._batch_size = batch_size
        self._n_insts = n_insts
        self._src_word2idx = src_word2idx
        self._tgt_word2idx = tgt_word2idx
        self._dataset = _Instances(image_dir, transform, image_insts or None, src_insts or None, tgt_insts or None,
                                   seed)
        self._collate = _Collate(tgt_format)
        self._num_workers = num_workers if image_insts else 0
        self._batches = batches
        self._seed = seed
        self._epoch = 0
        self._order = list(range(n_insts))
        self._need_shuffle = shuffle and batches is None
        if self._need_shuffle and seed is None:
            random.shuffle(self._order)

    @property
    def n_insts(self):
        return self._n_insts

    @property
    def src_word2idx(self):
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        return self._tgt_word2idx

    def __len__(self):
        if self._batches is not None:
            return len(self._batches)
        return int(math.ceil(self._n_insts / float(self._batch_size)))

    def _epoch_order(self):
        if self._need_shuffle and self._seed is not None:
            return np.random.RandomState([self._seed, self._epoch]).permutation(self._n_insts).tolist()
        return self._order

    def __iter__(self):
        batches = self._batches
        if batches is None:
            order = self._epoch_order()
            batches = [order[i:i + self._batch_size] for i in range(0, self._n_insts, self._batch_size)]
        self._dataset.epoch = self._epoch
        if self._num_workers > 0:
            seeded = self._seed is not None
            seed_workers = {'worker_init_fn': functools.partial(_seed_worker, self._seed, self._epoch)}
            loader = torch.utils.data.DataLoader(
                self._dataset, batch_sampler=batches, collate_fn=self._collate, num_workers=self._num_workers,
                pin_memory=self.cuda, **(seed_workers if seeded else {}))
            if seeded:
                # Starting the workers draws their seeds from PyTorch's random number generator. Restore its
                # state, so that the numbers the training draws do not depend on the number of workers.
                state = torch.get_rng_state()
                loader = iter(loader)
                torch.set_rng_state(state)
        else:  # in this process, like the original code
            loader = (self._collate([self._dataset[i] for i in batch]) for batch in batches)
        for indices, image, src, tgt in loader:
            if self.cuda:
                image = image.cuda(non_blocking=True) if image is not None else None
                src = src.cuda(non_blocking=True) if src is not None else None
                tgt = tgt.cuda(non_blocking=True) if tgt is not None else None
            yield indices, image, src, tgt
        self._epoch += 1
        if self._need_shuffle and self._seed is None:
            random.shuffle(self._order)
