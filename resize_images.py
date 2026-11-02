''' Resizes the images of a directory in advance, as the crops center_crop and resize_random_crop of train.py
and translate.py resize them first: the shorter side to 256 for -crop_size 224, keeping the aspect ratio. The
resized images are saved without loss (PNG) in another directory, with resized.json. Given as -image_dir, that
directory gives these crops the same images as the original directory, and reading them is much faster than
decoding and resizing full-size photos. The results are the same with the Pillow and torchvision versions that
resized the images (resized.json records them; train.py and translate.py warn about other versions, which can
decode JPEG images slightly differently). The other crops (random_crop, random_resized_crop) need the original
images.

    python resize_images.py -input KAGGLE/images -output KAGGLE/images_256 [-crop_size 224] [-workers 8]

The copy of IMAGE.jpg is IMAGE.png, in the same subdirectory. An image that cannot be read, resized or saved as
PNG (a CMYK JPEG image, for example) gets no copy and is listed in resized.json: training and prediction then
use a zero image for it, as they do when they cannot read or transform the original (a CMYK image has four
channels).

The copies are written into OUTPUT.partial, each under a temporary name until it is complete, and with
resized.json at the end the directory is renamed to OUTPUT. So OUTPUT exists only when all copies do, and
train.py and translate.py stop if -image_dir does not exist: no run reads an unfinished directory. An
interrupted run can be started again: it continues in OUTPUT.partial.
'''

import argparse
import io
import json
import os
import sys
import time
from multiprocessing import Pool

import PIL
import torchvision
from PIL import Image
from torchvision.transforms import functional as F

from DataLoader import RESIZED_RECORD, resize_size, resized_name


def resize(job):
    ''' Writes the resized copy of an image if it does not exist; returns None, or why the image has no copy. '''
    source, target, size = job
    if os.path.exists(target):
        return None
    partial = target + '.part'
    try:
        image = F.resize(Image.open(source), size)  # as transforms.Resize, the first step of the crops
        image.save(partial, 'PNG')
        os.rename(partial, target)
        return None
    except Exception as e:
        if os.path.exists(partial):
            os.remove(partial)
        return '{}: {}'.format(type(e).__name__, e)


def available_cpus():
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-input', required=True, help='the directory of the original images')
    parser.add_argument('-output', required=True, help='the directory of the resized images')
    parser.add_argument('-crop_size', type=int, default=224, help='the -crop_size of training and prediction')
    parser.add_argument('-workers', type=int, default=available_cpus(),
                        help='processes that resize images (default: the available CPUs)')
    opt = parser.parse_args()
    source_dir, target_dir = os.path.abspath(opt.input), os.path.abspath(opt.output)
    if target_dir == source_dir or target_dir.startswith(source_dir + os.sep):
        parser.error('-output must be outside -input')
    size = resize_size(opt.crop_size)
    if os.path.exists(target_dir):
        record_path = os.path.join(target_dir, RESIZED_RECORD)
        if not os.path.exists(record_path):
            parser.error('{} exists and is not a directory that resize_images.py completed'.format(target_dir))
        with io.open(record_path, encoding='utf-8') as f:
            record = json.load(f)
        if record['shorter_side'] != size or record['source'] != source_dir:
            parser.error('{} holds the images of {} resized to {}'.format(
                target_dir, record['source'], record['shorter_side']))
        print('{} is complete: {} images resized to {}'.format(target_dir, record['images'], size))
        return
    staging = target_dir + '.partial'
    settings = {'source': source_dir, 'shorter_side': size}
    settings_path = os.path.join(staging, 'partial.json')
    if os.path.exists(settings_path):  # continue an interrupted run with the same settings
        with io.open(settings_path, encoding='utf-8') as f:
            if json.load(f) != settings:
                parser.error('{} holds copies of other settings: remove it'.format(staging))
    else:
        if not os.path.isdir(staging):
            os.makedirs(staging)
        with io.open(settings_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(settings, sort_keys=True) + u'\n')

    names = []
    for root, dirs, files in os.walk(source_dir):
        dirs.sort()
        names += [os.path.relpath(os.path.join(root, name), source_dir) for name in sorted(files)]
    copies = {}
    for name in names:
        if resized_name(name) in copies:
            parser.error('{} and {} would have the same copy {}'.format(copies[resized_name(name)], name,
                                                                        resized_name(name)))
        copies[resized_name(name)] = name
    for directory in sorted(set(os.path.dirname(copy) for copy in copies)):
        if not os.path.isdir(os.path.join(staging, directory)):
            os.makedirs(os.path.join(staging, directory))

    jobs = [(os.path.join(source_dir, name), os.path.join(staging, resized_name(name)), size) for name in names]
    without_copy = {}
    start = time.time()
    pool = Pool(max(1, opt.workers))
    try:
        for k, (name, error) in enumerate(zip(names, pool.imap(resize, jobs, chunksize=32))):
            if error is not None:
                without_copy[name] = error
            if (k + 1) % 5000 == 0 or k + 1 == len(names):
                print('{} of {} images, {:.0f} s'.format(k + 1, len(names), time.time() - start))
                sys.stdout.flush()
    finally:
        pool.close()
        pool.join()

    record = {'source': source_dir, 'shorter_side': size, 'images': len(names) - len(without_copy),
              'without_copy': without_copy, 'pillow': PIL.__version__, 'torchvision': torchvision.__version__}
    with io.open(os.path.join(staging, RESIZED_RECORD), 'w', encoding='utf-8') as f:
        f.write(json.dumps(record, indent=2, sort_keys=True) + u'\n')
    os.remove(settings_path)
    os.rename(staging, target_dir)  # the copies appear under -output only now, all of them
    print('{}: {} images resized to {} (shorter side), {} without a copy (listed in {})'.format(
        target_dir, record['images'], size, len(without_copy), RESIZED_RECORD))


if __name__ == '__main__':
    main()
