''' Batches of images, titles and labels for training and prediction.

The batches are those of the original code: consecutive slices of the data, which is shuffled with Python's
random.shuffle when the loader is created and after every epoch; titles and token sequences are padded to the
longest in the batch. Images are read and transformed in worker processes (num_workers > 0), so that loading
overlaps with the computation on the GPU.
'''

import math
import os
import random

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms

import rnmtplus.Constants as Constants

# The target formats: token sequences (sequential), class indices (multiclass), one index per attribute.
TGT_FORMATS = ('sequence', 'chain', 'attributes')
CROPS = {'random_crop': 'RandomCrop', 'random_resized_crop': 'RandomResizedCrop'}
ZERO_IMAGE_SIZE = 224  # the size of the zero image that replaces a missing or unreadable image


def image_transform(crop, crop_size=224):
    ''' The image preprocessing of the experiments, for training, validation and prediction alike: a random
    crop (RandomCrop or RandomResizedCrop, depending on the experiment) and a random horizontal flip. '''
    return transforms.Compose([
        getattr(transforms, CROPS[crop])(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


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

    def __init__(self, image_dir, transform, image_insts, src_insts, tgt_insts):
        self.image_dir = image_dir
        self.transform = transform
        self.image_insts = image_insts
        self.src_insts = src_insts
        self.tgt_insts = tgt_insts

    def __len__(self):
        return len(self.src_insts if self.src_insts is not None else self.image_insts)

    def __getitem__(self, index):
        image = None
        if self.image_insts is not None:
            image = load_image(self.image_dir, self.image_insts[index], self.transform)
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
                 num_workers=0, batches=None):
        ''' image_insts, src_insts, tgt_insts: the instances of data.pt (preprocess.py); for the 'attributes'
        format, tgt_insts has one list per attribute. batches: lists of indices to use as the batches,
        instead of slices of batch_size (no shuffling). test: not used (kept from the original code). '''
        assert image_insts or src_insts
        assert tgt_format in TGT_FORMATS
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
        self._dataset = _Instances(image_dir, transform, image_insts or None, src_insts or None, tgt_insts or None)
        self._collate = _Collate(tgt_format)
        self._num_workers = num_workers if image_insts else 0
        self._batches = batches
        self._order = list(range(n_insts))
        self._need_shuffle = shuffle and batches is None
        if self._need_shuffle:
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

    def __iter__(self):
        batches = self._batches
        if batches is None:
            batches = [self._order[i:i + self._batch_size] for i in range(0, self._n_insts, self._batch_size)]
        if self._num_workers > 0:
            loader = torch.utils.data.DataLoader(
                self._dataset, batch_sampler=batches, collate_fn=self._collate, num_workers=self._num_workers,
                pin_memory=self.cuda)
        else:  # in this process, like the original code
            loader = (self._collate([self._dataset[i] for i in batch]) for batch in batches)
        for indices, image, src, tgt in loader:
            if self.cuda:
                image = image.cuda(non_blocking=True) if image is not None else None
                src = src.cuda(non_blocking=True) if src is not None else None
                tgt = tgt.cuda(non_blocking=True) if tgt is not None else None
            yield indices, image, src, tgt
        if self._need_shuffle:
            random.shuffle(self._order)
