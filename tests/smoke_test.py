''' A quick end-to-end check with synthetic data: for every settings file in experiments/ (or the given ones),
run.py vocab, preprocess, train (two epochs of a smaller model), translate and evaluate. The settings in
experiments/kaggle_seeded/ get a synthetic styles.csv and images/ and run train (-seed 1), translate and evaluate.

    python tests/smoke_test.py [SETTINGS ...] [-keep DIR] [-no_cuda]

The ImageNet weights of ResNet-50 are downloaded on first use. Prints one line per settings file and exits
with status 1 if any of them fails.
'''

import argparse
import glob
import io
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS = ['red', 'blue', 'black', 'white', 'shirt', 'dress', 'men', 'women', 'cotton', 'slim', 'coat', 'skirt',
         'jeans', 'casual', 'summer', 'winter', 'shoe', 'bag', 'long', 'short']
ATTRIBUTES = {
    'rakuten': [['Male', 'Female'], ['Upper', 'Lower'], ['Topwear', 'Dresses'], ['Top', 'Coat', 'Skirt'],
                ['Shirt', 'Blouse', 'Anorak'], ['Slim', 'Regular'], ['Normal', 'Stretchy']],
    'kaggle': [['Men', 'Women'], ['Apparel', 'Footwear'], ['Topwear', 'Shoes'], ['Tshirts', 'Shirts', 'Casual_Shoes'],
               ['Navy_Blue', 'Black', 'White'], ['Summer', 'Winter'], ['2011', '2012'], ['Casual', 'Formal']]}
AMAZON = [['Clothing', 'Women', 'Dresses'], ['Clothing', 'Men', 'Shirts', 'Casual'], ['Shoes', 'Women', 'Sandals'],
          ['Jewelry', 'Rings'], ['Clothing', 'Girls', 'Dresses'], ['Shoes', 'Men', 'Boots', 'Leather']]
SIZES = {'train': 24, 'valid': 8, 'test': 8}
KAGGLE_VALUES = [['Men', 'Women', 'Unisex'], ['Apparel', 'Footwear'], ['Topwear', 'Shoes'],
                 ['Tshirts', 'Casual Shoes', 'Shirts'], ['Navy Blue', 'Black', 'White'], ['Summer', 'Winter', ''],
                 ['2011', '2012'], ['Casual', 'Formal']]
# For the settings in experiments/kaggle_seeded: the split sizes and a SentencePiece model for the small data.
KAGGLE_OPTIONS = ['-seed', '1', '-valid_size', str(SIZES['valid']), '-test_size', str(SIZES['test']),
                  '-sp_vocab_size', '40']


def write(path, lines):
    with io.open(path, 'w', encoding='utf-8') as f:
        f.write(''.join(line + '\n' for line in lines))


def make_kaggle_data(root, rng):
    ''' A synthetic styles.csv and images/ in the format of the Kaggle data: some titles with commas, some
    empty values, a product without image. '''
    os.makedirs(os.path.join(root, 'images'))
    header = 'id,gender,masterCategory,subCategory,articleType,baseColour,season,year,usage,productDisplayName'
    rows = [header]
    for k in range(sum(SIZES.values())):
        product_id = str(1000 + k)
        title = ' '.join(rng.choice(WORDS) for _ in range(rng.randint(3, 8)))
        if k % 5 == 0:
            title += ', ' + rng.choice(WORDS)
        rows.append(','.join([product_id] + [rng.choice(v) for v in KAGGLE_VALUES] + [title]))
        if k != 3:
            a = np.full((260, 200, 3), rng.randrange(256), dtype=np.uint8)
            a[rng.randrange(130):, rng.randrange(100):] = [rng.randrange(256) for _ in range(3)]
            Image.fromarray(a).save(os.path.join(root, 'images', product_id + '.jpg'))
    write(os.path.join(root, 'styles.csv'), rows)


def make_data(root):
    ''' Synthetic data in the file formats of the three datasets; root/images holds the images, root/kaggle_raw
    the synthetic Kaggle data for experiments/kaggle_seeded. '''
    rng = random.Random(0)
    make_kaggle_data(os.path.join(root, 'kaggle_raw'), rng)
    os.makedirs(os.path.join(root, 'images'))
    for k in range(10):
        a = np.full((256, 256, 3), rng.randrange(256), dtype=np.uint8)
        a[rng.randrange(128):, rng.randrange(128):] = [rng.randrange(256) for _ in range(3)]
        Image.fromarray(a).save(os.path.join(root, 'images', '%d.jpg' % k))
    title = lambda: ' '.join(rng.choice(WORDS) for _ in range(rng.randint(3, 8)))
    image = lambda: '%d.jpg' % rng.randrange(10)
    for ds, attrs in ATTRIBUTES.items():
        d = os.path.join(root, ds)
        os.makedirs(d)
        for split, n in SIZES.items():
            titles = [title() for _ in range(n)]
            write(os.path.join(d, split + '.img'), [image() for _ in range(n)])
            write(os.path.join(d, split + '.tit'), titles)
            write(os.path.join(d, split + '.enc.tit'), titles)
            write(os.path.join(d, split + '.lbl'), [' '.join(rng.choice(v) for v in attrs) for _ in range(n)])
    d = os.path.join(root, 'amazon')
    os.makedirs(d)
    for split, n in SIZES.items():
        products = [(image(), title(), [' '.join(c) for c in rng.sample(AMAZON, rng.randint(1, 2))]) for _ in range(n)]
        if split != 'test':  # one line per (product, chain)
            rows = [(img, tit, chain) for img, tit, chains in products for chain in chains]
            for ext, k in (('img', 0), ('tit', 1), ('lbl', 2)):
                write(os.path.join(d, 'e%s.%s' % (split, ext)), [r[k] for r in rows])
        if split != 'train':
            write(os.path.join(d, split + '.img'), [p[0] for p in products])
            write(os.path.join(d, split + '.enc.tit'), [p[1] for p in products])
            write(os.path.join(d, split + '.cat'), [' ||| '.join(p[2]) for p in products])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('settings', nargs='*', help='default: experiments/*/*/*.json')
    parser.add_argument('-keep', metavar='DIR', help='keep the data and the outputs in DIR')
    parser.add_argument('-no_cuda', action='store_true')
    opt = parser.parse_args()
    settings = opt.settings or sorted(glob.glob(os.path.join(REPO, 'experiments', '*', '*', '*.json')))
    root = opt.keep or tempfile.mkdtemp(prefix='smoke_test_')
    make_data(root)
    cuda = ['-no_cuda'] if opt.no_cuda else []
    failures = 0
    for path in settings:
        dataset = os.path.basename(os.path.dirname(os.path.dirname(path)))
        name = os.path.relpath(path, os.path.join(REPO, 'experiments'))
        work = os.path.join(root, 'work', name.replace('.json', ''))
        log = os.path.join(root, 'work', name.replace('.json', '.log'))
        os.makedirs(work)
        train = ['-epoch', '2', '-n_layers', '3', '-batch_size', '4', '-save_mode', 'best', '-patience', '1',
                 '-num_workers', '2'] + cuda
        translate = ['-checkpoint', os.path.join(work, 'models', 'model.best.chkpt'), '-batch_size', '4',
                     '-num_workers', '2'] + cuda
        if dataset == 'kaggle_seeded':
            base = [sys.executable, os.path.join(REPO, 'run.py'), path, None,
                    '-data_dir', os.path.join(root, 'kaggle_raw'), '-work_dir', work]
            stages = [('train', train + KAGGLE_OPTIONS), ('translate', translate), ('evaluate', [])]
        else:  # experiments/<dataset>_fixed: the data of <dataset>
            data_dir = os.path.join(root, dataset[:-len('_fixed')] if dataset.endswith('_fixed') else dataset)
            base = [sys.executable, os.path.join(REPO, 'run.py'), path, None, '-data_dir', data_dir,
                    '-image_dir', os.path.join(root, 'images'), '-work_dir', work]
            stages = [('vocab', []), ('preprocess', []), ('train', train + ['-seed', '1']), ('translate', translate),
                      ('evaluate', [])]
        start = time.time()
        status = 'ok'
        with open(log, 'w') as f:
            for stage, extra in stages:
                command = base[:3] + [stage] + base[4:] + extra
                f.write('$ ' + ' '.join(command) + '\n')
                f.flush()
                if subprocess.call(command, stdout=f, stderr=subprocess.STDOUT):
                    status = 'FAILED in %s (see %s)' % (stage, log)
                    break
        if status == 'ok':
            with io.open(os.path.join(work, 'pred_test.txt'), encoding='utf-8') as f:
                if len(f.read().splitlines()) != SIZES['test']:
                    status = 'FAILED: wrong number of predictions'
        failures += status != 'ok'
        print('%-34s %-5s %5.1f s' % (name, status, time.time() - start))
        sys.stdout.flush()
    if not opt.keep:
        shutil.rmtree(root)
    print('%d of %d failed' % (failures, len(settings)) if failures else 'all %d passed' % len(settings))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
