''' The FashionGen data (the export of fashiongen_export.py: products.csv and images/), with its fixed test set.

train.py -fashiongen_dir DIR -seed N calls prepare(), which prepares the data directory of the experiment with the
steps and files of kaggle_data.py: SentencePiece model, vocabularies and data.pt (split.json records the
checksum of products.csv). The split is the release's: the products of its validation file (7,519) are the
test set, and the seed draws the validation set (-valid_size, 5,000) from the products of its training file, the
rest of which (55,147) is the training set. For each product:
- the text is its name and description as they are in the release, "<name>. <description>", or the name alone
  if the description is empty; only a line break in it (in 594 descriptions, mostly \r\n) becomes a space,
  since the data files hold one product per line;
- the attribute chain holds its 6 attributes in a coarse-to-fine order (ATTRIBUTES), with the spaces in a value
  replaced by "_", as for the Kaggle data;
- the image is its views: one name per view slot, in the order of the poses id_gridfs_1 to id_gridfs_6 and
  separated by tabs (DataLoader.VIEW_SEPARATOR), <product_id>/<pose>.png, or None for a view the product does
  not have. With train.py -max_views 6, the
  models encode every view and attend to (or average) the regions of all of them, with a learned embedding of
  each slot (DataLoader.py, rnmtplus/Models.py).
The release has no missing values.

The order of the attributes, found as for the H&M data (hm_data.py): if one attribute determines another, the
determined one comes first (the category determines the department); of the orders that follow this rule, the
chain is the one whose prefixes add the least information (the sum of their entropies over the products of the
release's training file is the lowest). The same order results on two random halves of these products and on all
products; gender before department would add 0.02 bits.

To prepare the data without training:

    python fashiongen_data.py -fashiongen_dir DIR -model_type TYPE -seed N -output_dir OUT
'''

import argparse
import csv
import io
import os
import re

import kaggle_data
from DataLoader import VIEW_SEPARATOR
from kaggle_data import Product
from rnmtplus import compat
from rnmtplus.Models import MODEL_TYPES

# The columns of the attribute chain, coarse to fine: department (4 values), gender (2), category (48),
# subcategory (121), season (10) and brand (570).
ATTRIBUTES = ('department', 'gender', 'category', 'subcategory', 'season', 'brand')
COLUMNS = ('product_id', 'split', 'name', 'description', 'poses') + ATTRIBUTES
POSES = tuple('id_gridfs_{}'.format(k) for k in range(1, 7))  # the view slots, in order
MAX_VIEWS = len(POSES)
TEST_FILE = 'validation'  # the file of the release whose products are the test set
LINE_BREAK = re.compile(r'\r\n|\r|\n')  # one line break, each of which becomes a space in the text


def image_files(fashiongen_dir):
    ''' The image files under fashiongen_dir/images, as paths relative to it. '''
    root = os.path.join(fashiongen_dir, 'images')
    found = set()
    for directory, _, names in os.walk(root):
        found.update(os.path.relpath(os.path.join(directory, name), root).replace(os.sep, '/') for name in names)
    return found


def read_products(fashiongen_dir):
    ''' The products of products.csv, in the order of the file, and the ids of the test set. '''
    images = image_files(fashiongen_dir)
    with io.open(os.path.join(fashiongen_dir, 'products.csv'), encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        absent = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if absent:
            raise ValueError('products.csv has no column ' + ', '.join(absent))
        products, test_ids = [], []
        for row in reader:
            product = row['product_id']
            poses = row['poses'].split()
            unknown = [p for p in poses if p not in POSES]
            if unknown:
                raise ValueError('product {} has the unknown views {}'.format(product, ' '.join(unknown)))
            names = ['{}/{}.png'.format(product, pose) for pose in POSES]
            views = VIEW_SEPARATOR.join(name if pose in poses and name in images else 'None'
                                        for pose, name in zip(POSES, names))
            text = LINE_BREAK.sub(' ', row['name'] + ('. ' + row['description'] if row['description'] else ''))
            chain = ' '.join(row[c].replace(' ', '_') if row[c] else 'None' for c in ATTRIBUTES)
            products.append(Product(product, views, text, chain))
            if row['split'] == TEST_FILE:
                test_ids.append(product)
    return products, test_ids


def prepare(fashiongen_dir, output_dir, model_type, seed, valid_size=5000, sp_vocab_size=4000, distinct_unk=False,
            missing_values='', exclude_ids=''):
    ''' Writes the data directory of an experiment (see above and kaggle_data.py) and returns the path of its
    data.pt. A directory that was prepared with other settings is not overwritten. '''
    if not 0 <= seed < 2 ** 32:
        raise ValueError('the seed must be between 0 and 2**32 - 1')
    checksum = ('products_csv_md5', kaggle_data.file_md5(os.path.join(fashiongen_dir, 'products.csv')))
    products, test_ids = read_products(fashiongen_dir)
    return kaggle_data.prepare_products(
        products, checksum, fashiongen_dir, output_dir, model_type, seed, valid_size=valid_size,
        sp_vocab_size=sp_vocab_size, distinct_unk=distinct_unk, missing_values=missing_values,
        exclude_ids=exclude_ids, test_ids=test_ids, test_set="the products of the release's validation file")


def main():
    parser = argparse.ArgumentParser(description='Prepares the data directory of an experiment on the FashionGen '
                                                 'data (its fixed test set, the validation set drawn with -seed), '
                                                 'as train.py -fashiongen_dir does.')
    parser.add_argument('-fashiongen_dir', required=True,
                        help='the FashionGen data as fashiongen_export.py writes it: products.csv and images/')
    parser.add_argument('-output_dir', required=True)
    parser.add_argument('-model_type', required=True, choices=MODEL_TYPES, help='sets the format of the targets')
    parser.add_argument('-seed', type=int, required=True)
    parser.add_argument('-valid_size', type=float, default=5000,
                        help='products in the validation set, drawn from those of the release\'s training file; '
                             'a fraction of them if below 1')
    parser.add_argument('-sp_vocab_size', type=int, default=4000, help='pieces of the SentencePiece model')
    parser.add_argument('-distinct_unk', action='store_true', help='build_vocab.py -distinct_unk')
    parser.add_argument('-missing_values', default='', help='comma-separated values of missing attributes')
    parser.add_argument('-exclude_ids', default='', help='comma-separated ids of products to leave out')
    opt = parser.parse_args()
    if not 0 <= opt.seed < 2 ** 32:
        parser.error('-seed must be between 0 and 2**32 - 1')
    compat.seed_python_hash(opt.seed)
    compat.seed_everything(opt.seed)
    prepare(opt.fashiongen_dir, opt.output_dir, opt.model_type, opt.seed, opt.valid_size, opt.sp_vocab_size,
            opt.distinct_unk, opt.missing_values, opt.exclude_ids)


if __name__ == '__main__':
    main()
