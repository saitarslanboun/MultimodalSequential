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

The image directory can also be one that resize_images.py wrote (it holds RESIZED_RECORD): the images resized
in advance as the crops center_crop and resize_random_crop resize them first, saved without loss. The loader
then reads the resized copy <name without extension>.png of an image, which these crops leave as it is, so the
images are those of the original directory (check_image_dir: only for these crops and their size).

With mask_missing_images (train.py -mask_missing_images), a product without an image (the name None, or an
image that cannot be read or transformed) is marked absent: the batches then hold MaskedImages (the images and
their presence), and the models leave out the image of an absent product, which keeps its text and targets.
Without it, such a product gets a zero image, as in the original code.

With max_views > 1 (train.py -max_views, FashionGen), a product has several views: its line of an .img file
names one image per view slot, in slot order and separated by tabs (a name can have spaces), or None for a view
it does not have (the slots after the last name are absent too). Its image is then max_views x 3 x size x
size, the batches always hold MaskedImages with the presence of every view (batch x views), and the models leave
out the views that are not present. With a seed, the random crop and flip of each view are drawn with a
random.Random seeded with the seed, the epoch, the index of the product and the slot.
'''

import functools
import io
import json
import math
import os
import random
import warnings

import numpy as np
import PIL
import torch
import torch.utils.data
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

import rnmtplus.Constants as Constants
from rnmtplus.Models import MaskedImages

# The target formats: token sequences (sequential), class indices (multiclass), one index per attribute.
TGT_FORMATS = ('sequence', 'chain', 'attributes')
CROPS = {'random_crop': 'RandomCrop', 'random_resized_crop': 'RandomResizedCrop', 'center_crop': 'CenterCrop',
         'resize_random_crop': 'RandomCrop'}
# The crops that first resize the shorter side of an image to resize_size(crop_size), keeping the aspect ratio:
# they can read the images that resize_images.py resized in advance.
RESIZING_CROPS = ('center_crop', 'resize_random_crop')
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # of ImageNet, for the ResNet
RESIZE_RATIO = 256 / 224.  # the shorter side 256 for a crop of 224, as for ImageNet
ZERO_IMAGE_SIZE = 224  # the size of the zero image that replaces a missing or unreadable image
RESIZED_RECORD = 'resized.json'  # written by resize_images.py into the directory of the resized images


def resize_size(crop_size):
    ''' The shorter side to which center_crop and resize_random_crop resize an image: 256 for a crop of 224. '''
    return int(round(crop_size * RESIZE_RATIO))


def image_transform(crop, crop_size=224):
    ''' The image preprocessing.
    random_crop, random_resized_crop: the random crop of the experiment (RandomCrop or RandomResizedCrop) and a
    random horizontal flip, which the paper's runs used in training, validation and prediction alike.
    resize_random_crop: resize the shorter side to 256/224 of the crop size (keeping the aspect ratio), then a
    random crop and a random horizontal flip. Unlike random_resized_crop, which can take 8% of the image, the
    crop always shows most of it.
    center_crop: resize the same way and crop the center; no randomness. '''
    if crop == 'center_crop':
        steps = [transforms.Resize(resize_size(crop_size)), transforms.CenterCrop(crop_size)]
    elif crop == 'resize_random_crop':
        steps = [transforms.Resize(resize_size(crop_size)), transforms.RandomCrop(crop_size),
                 transforms.RandomHorizontalFlip()]
    else:
        steps = [getattr(transforms, CROPS[crop])(crop_size), transforms.RandomHorizontalFlip()]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def resized_record(image_dir):
    ''' The record of resize_images.py if it wrote image_dir, else None. '''
    path = os.path.join(image_dir, RESIZED_RECORD) if image_dir else None
    if path is None or not os.path.isfile(path):
        return None
    with io.open(path, encoding='utf-8') as f:
        return json.load(f)


def check_image_dir(image_dir, crops, crop_size):
    ''' Raises ValueError if image_dir does not exist (all images would be zero images), and for a directory of
    resized images (resize_images.py) if one of the crops does not resize the images first to the size they
    have; warns if other Pillow or torchvision versions resized them, which can decode and resize the original
    images slightly differently. '''
    if not os.path.isdir(image_dir):
        raise ValueError('the image directory {} does not exist'.format(image_dir))
    record = resized_record(image_dir)
    if record is None:
        return
    others = sorted(set(crop for crop in crops if crop not in RESIZING_CROPS))
    if others:
        raise ValueError('{} holds images resized in advance (resize_images.py), which {} cannot use: it needs '
                         'the original images'.format(image_dir, ' and '.join(others)))
    if record['shorter_side'] != resize_size(crop_size):
        raise ValueError('the images in {} are resized to {}, the crops of {} resize to {}'.format(
            image_dir, record['shorter_side'], crop_size, resize_size(crop_size)))
    versions = {'pillow': PIL.__version__, 'torchvision': torchvision.__version__}
    other_versions = sorted(k for k, v in versions.items() if record.get(k) != v)
    if other_versions:
        warnings.warn('{} was resized with {}; this is {}: the images can differ slightly from those resized '
                      'here'.format(image_dir, ', '.join('{} {}'.format(k, record.get(k)) for k in other_versions),
                                    ', '.join('{} {}'.format(k, versions[k]) for k in other_versions)))


def resized_name(name):
    ''' The file name that resize_images.py gives the resized copy of the image file name. '''
    return os.path.splitext(name)[0] + '.png'


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
    RandomResizedCrop (area 8% to 100%, aspect ratio 3/4 to 4/3); resize_random_crop first resizes the image as
    image_transform does. '''

    def __init__(self, crop, crop_size=224):
        if crop not in ('random_crop', 'random_resized_crop', 'resize_random_crop'):
            raise ValueError('not a random crop: ' + crop)
        self.crop = crop
        self.crop_size = crop_size

    def __call__(self, img, rng):
        size = self.crop_size
        if self.crop == 'random_resized_crop':
            top, left, h, w = _resized_crop_params(img, rng)
            img = F.resized_crop(img, top, left, h, w, (size, size))
        else:
            if self.crop == 'resize_random_crop':
                img = F.resize(img, resize_size(size))  # as transforms.Resize in image_transform
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


def read_image(image_dir, img_inst, transform, resized=False):
    ''' The transformed image and whether it is present: a zero image and False if the name is "None" or the
    file cannot be read or transformed. resized: image_dir holds the resized copies of resize_images.py, which
    has none of an image that it could not read or save, so that this image too is a zero image. '''
    if type(img_inst) is list:  # names read with read_instances_from_file: [<BOS>, name, <EOS>]
        img_inst = img_inst[1]
    if img_inst == "None":
        return torch.zeros((3, ZERO_IMAGE_SIZE, ZERO_IMAGE_SIZE)), False
    try:
        return transform(Image.open(os.path.join(image_dir, resized_name(img_inst) if resized else img_inst))), True
    except Exception:
        return torch.zeros((3, ZERO_IMAGE_SIZE, ZERO_IMAGE_SIZE)), False


VIEW_SEPARATOR = '\t'  # between the view slots of a line of an .img file


def view_names(img_inst):
    ''' The image names of an image instance: a line of an .img file (data.pt, translate.py) split at its tabs,
    or the tokens of preprocess.read_instances_from_file without <BOS> and <EOS>. '''
    if type(img_inst) is list:
        return img_inst[1:-1]
    return img_inst.split(VIEW_SEPARATOR)


def read_views(image_dir, img_inst, view_transform, max_views, resized=False):
    ''' The transformed views of a product (max_views x 3 x size x size) and whether each is present: the names
    of img_inst are its view slots in order (None for a view it does not have), the slots after them are absent
    too, and an absent view is a zero image (read_image). view_transform(slot): the transformation of a slot. '''
    names = view_names(img_inst)
    if len(names) > max_views:
        raise ValueError('{} views, more than max_views {}: {}'.format(len(names), max_views, ' '.join(names)))
    views, present = [], []
    for slot in range(max_views):
        image, ok = read_image(image_dir, names[slot] if slot < len(names) else 'None', view_transform(slot), resized)
        views.append(image)
        present.append(ok)
    return torch.stack(views), present


def load_image(image_dir, img_inst, transform, resized=False):
    ''' The transformed image; a zero image if the name is "None" or the file cannot be read or transformed. '''
    return read_image(image_dir, img_inst, transform, resized)[0]


def pad_to_longest(insts):
    ''' The token index lists as a LongTensor, padded with PAD to the longest. '''
    max_len = max(len(inst) for inst in insts)
    return torch.LongTensor([inst + [Constants.PAD] * (max_len - len(inst)) for inst in insts])


class _Instances(torch.utils.data.Dataset):

    def __init__(self, image_dir, transform, image_insts, src_insts, tgt_insts, seed=None, max_views=1):
        self.image_dir = image_dir
        self.resized = image_insts is not None and resized_record(image_dir) is not None
        self.transform = transform
        self.image_insts = image_insts
        self.src_insts = src_insts
        self.tgt_insts = tgt_insts
        self.seed = seed
        self.max_views = max_views
        self.epoch = 0  # set by the DataLoader before each epoch

    def _view_transform(self, index, slot):
        ''' The transformation of a view of the product: seeded with the seed, the epoch, the index and the slot. '''
        if isinstance(self.transform, _SeededTransform):
            rng = random.Random('{} {} {} {}'.format(self.seed, self.epoch, index, slot))
            return functools.partial(self.transform, rng=rng)
        return self.transform

    def __len__(self):
        return len(self.src_insts if self.src_insts is not None else self.image_insts)

    def __getitem__(self, index):
        image = present = None
        if self.image_insts is not None and self.max_views > 1:
            image, present = read_views(self.image_dir, self.image_insts[index],
                                        functools.partial(self._view_transform, index), self.max_views, self.resized)
        elif self.image_insts is not None:
            transform = self.transform
            if isinstance(transform, _SeededTransform):
                rng = random.Random('{} {} {}'.format(self.seed, self.epoch, index))
                transform = functools.partial(transform, rng=rng)
            image, present = read_image(self.image_dir, self.image_insts[index], transform, self.resized)
        src = self.src_insts[index] if self.src_insts is not None else None
        tgt = self.tgt_insts[index] if self.tgt_insts is not None else None
        return index, image, present, src, tgt


class _Collate(object):

    def __init__(self, tgt_format, mask_missing_images=False, max_views=1):
        self.tgt_format = tgt_format
        self.mask_missing_images = mask_missing_images
        self.max_views = max_views

    def __call__(self, items):
        indices, images, presents, srcs, tgts = zip(*items)
        image = torch.stack(images) if images[0] is not None else None
        # one value per product, or per view (batch x views); uint8 rather than bool, which PyTorch 1.1 does not have
        masked = image is not None and (self.mask_missing_images or self.max_views > 1)
        present = torch.tensor(presents, dtype=torch.uint8) if masked else None
        src = pad_to_longest(srcs) if srcs[0] is not None else None
        if tgts[0] is None:
            tgt = None
        elif self.tgt_format == 'sequence':
            tgt = pad_to_longest(tgts)
        else:
            tgt = torch.LongTensor(list(tgts))
        return list(indices), image, present, src, tgt


class DataLoader(object):
    ''' Iterating over it gives the batches of an epoch as tuples (indices, image, src, tgt): the indices of the
    products in the data, the images (batch x 3 x 224 x 224; with mask_missing_images, MaskedImages of them and
    their presence; with max_views > 1, MaskedImages of batch x views x 3 x 224 x 224 images and the presence
    of the views), the padded titles (batch x length) and the targets (a padded batch x length tensor for
    'sequence', batch for 'chain', batch x attributes for 'attributes'); None for what the data does not
    have. '''

    def __init__(self, transform, image_dir, src_word2idx, tgt_word2idx, image_insts=None, src_insts=None,
                 tgt_insts=None, tgt_format='sequence', cuda=True, batch_size=64, shuffle=True, test=False,
                 num_workers=0, batches=None, seed=None, mask_missing_images=False, max_views=1):
        ''' image_insts, src_insts, tgt_insts: the instances of data.pt (preprocess.py); for the 'attributes'
        format, tgt_insts has one list per attribute. batches: lists of indices to use as the batches,
        instead of slices of batch_size (no shuffling). seed: shuffle and transform the images with this seed
        (see above) instead of the global random number generators. mask_missing_images: mark the products
        without an image (see above). max_views: the view slots of a product (see above). test: not used (kept
        from the original code). '''
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
                                   seed, max_views)
        self._collate = _Collate(tgt_format, mask_missing_images, max_views)
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
        for indices, image, present, src, tgt in loader:
            if self.cuda:
                image = image.cuda(non_blocking=True) if image is not None else None
                present = present.cuda(non_blocking=True) if present is not None else None
                src = src.cuda(non_blocking=True) if src is not None else None
                tgt = tgt.cuda(non_blocking=True) if tgt is not None else None
            if present is not None:
                image = MaskedImages(image, present)
            yield indices, image, src, tgt
        self._epoch += 1
        if self._need_shuffle and self._seed is None:
            random.shuffle(self._order)
