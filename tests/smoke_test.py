''' A quick end-to-end check with synthetic data: for every settings file in experiments/ (or the given ones),
run.py vocab, preprocess, train (two epochs of a smaller model), translate and evaluate. The settings in
experiments/kaggle_seeded/ get a synthetic styles.csv and images/, those in experiments/hm_seeded/ a synthetic
articles.csv and images/, and run train (-seed 1), translate and evaluate.

    python tests/smoke_test.py [SETTINGS ...] [-keep DIR] [-no_cuda]

The ImageNet weights of ResNet-50 are downloaded on first use. Prints one line per settings file and exits
with status 1 if any of them fails.
'''

import argparse
import csv
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
               ['Navy_Blue', 'Black', 'White'], ['Summer', 'Winter', 'None'], ['2011', '2012'],
               ['Casual', 'Formal', 'NA']]}  # NA, None: the missing values of the Kaggle data
AMAZON = [['Clothing', 'Women', 'Dresses'], ['Clothing', 'Men', 'Shirts', 'Casual'], ['Shoes', 'Women', 'Sandals'],
          ['Jewelry', 'Rings'], ['Clothing', 'Girls', 'Dresses'], ['Shoes', 'Men', 'Boots', 'Leather']]
SIZES = {'train': 24, 'valid': 8, 'test': 8}
KAGGLE_VALUES = [['Men', 'Women', 'Unisex'], ['Apparel', 'Footwear'], ['Topwear', 'Shoes'],
                 ['Tshirts', 'Casual Shoes', 'Shirts'], ['Navy Blue', 'Black', 'White'], ['Summer', 'Winter', ''],
                 ['2011', '2012'], ['Casual', 'Formal', 'NA']]
# For the settings in experiments/kaggle_seeded and hm_seeded: the split sizes and a SentencePiece model for the
# small data.
KAGGLE_OPTIONS = ['-seed', '1', '-valid_size', str(SIZES['valid']), '-test_size', str(SIZES['test']),
                  '-sp_vocab_size', '40']
HM_HEADER = ['article_id', 'product_code', 'prod_name', 'product_type_no', 'product_type_name', 'product_group_name',
             'graphical_appearance_no', 'graphical_appearance_name', 'colour_group_code', 'colour_group_name',
             'perceived_colour_value_id', 'perceived_colour_value_name', 'perceived_colour_master_id',
             'perceived_colour_master_name', 'department_no', 'department_name', 'index_code', 'index_name',
             'index_group_no', 'index_group_name', 'section_no', 'section_name', 'garment_group_no',
             'garment_group_name', 'detail_desc']
# The (code, name) choices of the attributes of the synthetic H&M data: Unknown, Undefined and undefined are the
# missing values of the H&M data, and Jersey is a department name with two codes, as in articles.csv.
HM_VALUES = [('product_type', [('253', 'Vest top'), ('272', 'Trousers'), ('-1', 'Unknown')]),
             ('product_group', [(None, 'Garment Upper body'), (None, 'Garment Lower body'), (None, 'Unknown')]),
             ('graphical_appearance', [('1010016', 'Solid'), ('1010017', 'Stripe'), ('-1', 'Unknown')]),
             ('colour_group', [('09', 'Black'), ('73', 'Dark Blue'), ('90', 'Other')]),
             ('perceived_colour_value', [('4', 'Dark'), ('3', 'Light'), ('6', 'Undefined')]),
             ('perceived_colour_master', [('5', 'Black'), ('2', 'Blue'), ('14', 'undefined')]),
             ('department', [('1641', 'Jersey'), ('1660', 'Jersey'), ('1338', 'Trouser')]),
             ('index', [('A', 'Ladieswear'), ('F', 'Menswear')]),
             ('index_group', [('1', 'Ladieswear'), ('3', 'Menswear')]),
             ('section', [('16', 'Womens Everyday Basics'), ('26', 'Mens Underwear')]),
             ('garment_group', [('1002', 'Jersey Basic'), ('1001', 'Unknown')])]


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


def make_hm_data(root, rng):
    ''' A synthetic articles.csv and images/ in the format of the H&M data: descriptions with commas and quotes,
    some empty, names with two spaces, articles without an image and a grayscale image. '''
    os.makedirs(os.path.join(root, 'images'))
    with io.open(os.path.join(root, 'articles.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HM_HEADER)
        for k in range(sum(SIZES.values())):
            article = '0%09d' % (108775015 + 17 * k)
            row = {'article_id': article, 'product_code': article[:7]}
            for attribute, choices in HM_VALUES:
                code, name = rng.choice(choices)
                if code is not None:
                    row[next(c for c in HM_HEADER if c.startswith(attribute) and c != attribute + '_name')] = code
                row[attribute + '_name'] = name
            words = [rng.choice(WORDS) for _ in range(rng.randint(1, 4))]
            row['prod_name'] = ('  ' if k % 6 == 0 else ' ').join(words)
            description = ' '.join(rng.choice(WORDS) for _ in range(rng.randint(4, 12)))
            if k % 5 == 0:
                description += ', "' + rng.choice(WORDS) + '" ' + rng.choice(WORDS)
            row['detail_desc'] = '' if k % 4 == 1 else description.capitalize() + '.'
            writer.writerow([row[c] for c in HM_HEADER])
            if k % 7 == 3:
                continue  # no image
            os.makedirs(os.path.join(root, 'images', article[:3]), exist_ok=True)
            a = np.full((300, 200, 3), rng.randrange(256), dtype=np.uint8)
            a[rng.randrange(150):, rng.randrange(100):] = [rng.randrange(256) for _ in range(3)]
            image = Image.fromarray(a)
            if k == 5:
                image = image.convert('L')  # one grayscale image, as in the H&M data
            image.save(os.path.join(root, 'images', article[:3], article + '.jpg'))


def make_data(root):
    ''' Synthetic data in the file formats of the three datasets; root/images holds the images, root/kaggle_raw
    the synthetic Kaggle data for experiments/kaggle_seeded, root/hm_raw the synthetic H&M data for
    experiments/hm_seeded. '''
    rng = random.Random(0)
    make_kaggle_data(os.path.join(root, 'kaggle_raw'), rng)
    make_hm_data(os.path.join(root, 'hm_raw'), random.Random(1))
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
        if dataset in ('kaggle_seeded', 'hm_seeded'):
            raw = 'kaggle_raw' if dataset == 'kaggle_seeded' else 'hm_raw'
            base = [sys.executable, os.path.join(REPO, 'run.py'), path, None,
                    '-data_dir', os.path.join(root, raw), '-work_dir', work]
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
