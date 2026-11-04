''' The H&M Personalized Fashion Recommendations data (articles.csv and images/), split at random with a seed.

train.py -hm_dir DIR -seed N calls prepare(), which reads the articles of articles.csv and prepares the data
directory of the experiment with the steps and files of kaggle_data.py: the same split, SentencePiece model,
vocabularies and data.pt (split.json records the checksum of articles.csv). For each article:
- the text is its name and description as they are in articles.csv, "<prod_name>. <detail_desc>", or the name
  alone if the description is empty (416 articles);
- the attribute chain holds the names of the 11 attributes in a coarse-to-fine order (ATTRIBUTES), with the
  spaces in a name replaced by "_", as for the Kaggle data. A few names stand for several H&M codes (22
  department names, the section Ladies Other, the product type Umbrella); the attributes before them in the
  chain tell them apart;
- the image is images/<first three digits of the article id>/<article id>.jpg, or None for an article without
  an image (442): with train.py -mask_missing_images, the models then leave out the image of these articles
  (DataLoader.py), which stay in the data with their text.
The articles are split at random one by one, like the Kaggle products: the colour variants of a product can be
in different sets.

With missing_values (Unknown,Undefined,undefined: the placeholders of articles.csv), these names mark a missing
attribute, as NA and None do for the Kaggle data: they are not part of the target vocabularies, and data.pt has
-1 for them, which the losses ignore.

The order of the attributes: if one attribute determines another (the index its index group, the department its
index, section and garment group, the product type its product group), the determined one comes first; of the
orders that follow this rule, the chain is the one whose prefixes add the least information (the sum of their
entropies over the articles is the lowest). The same order results on two random halves of the products, with
the names instead of the codes, and without the articles that have a placeholder.

To prepare the data without training:

    python hm_data.py -hm_dir DIR -model_type TYPE -seed N -output_dir OUT
'''

import argparse
import csv
import io
import os

import kaggle_data
from kaggle_data import Product
from rnmtplus import compat
from rnmtplus.Models import MODEL_TYPES

# The columns of the attribute chain, coarse to fine: index group, index, product group, garment group, section,
# department, product type, lightness, perceived colour, colour and pattern.
ATTRIBUTES = ('index_group_name', 'index_name', 'product_group_name', 'garment_group_name', 'section_name',
              'department_name', 'product_type_name', 'perceived_colour_value_name', 'perceived_colour_master_name',
              'colour_group_name', 'graphical_appearance_name')
COLUMNS = ('article_id', 'prod_name', 'detail_desc') + ATTRIBUTES


def image_files(hm_dir):
    ''' The image files under hm_dir/images, as paths relative to it. '''
    root = os.path.join(hm_dir, 'images')
    found = set()
    for directory, _, names in os.walk(root):
        found.update(os.path.relpath(os.path.join(directory, name), root) for name in names)
    return found


def read_products(hm_dir):
    ''' The articles of articles.csv, in the order of the file. '''
    images = image_files(hm_dir)
    with io.open(os.path.join(hm_dir, 'articles.csv'), encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        absent = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if absent:
            raise ValueError('articles.csv has no column ' + ', '.join(absent))
        products = []
        for row in reader:
            article = row['article_id']
            image = '{}/{}.jpg'.format(article[:3], article)
            text = row['prod_name'] + ('. ' + row['detail_desc'] if row['detail_desc'] else '')
            chain = ' '.join(row[c].replace(' ', '_') if row[c] else 'None' for c in ATTRIBUTES)
            products.append(Product(article, image if image in images else 'None', text, chain))
    return products


def prepare(hm_dir, output_dir, model_type, seed, valid_size=5000, test_size=5000, sp_vocab_size=4000,
            distinct_unk=False, missing_values='', exclude_ids=''):
    ''' Writes the data directory of an experiment (see above and kaggle_data.py) and returns the path of its
    data.pt. A directory that was prepared with other settings is not overwritten. '''
    if not 0 <= seed < 2 ** 32:
        raise ValueError('the seed must be between 0 and 2**32 - 1')
    checksum = ('articles_csv_md5', kaggle_data.file_md5(os.path.join(hm_dir, 'articles.csv')))
    return kaggle_data.prepare_products(read_products(hm_dir), checksum, hm_dir, output_dir, model_type, seed,
                                        valid_size, test_size, sp_vocab_size, distinct_unk, missing_values,
                                        exclude_ids)


def main():
    parser = argparse.ArgumentParser(description='Splits the H&M data with a seed and prepares the data '
                                                 'directory of an experiment, as train.py -hm_dir does.')
    parser.add_argument('-hm_dir', required=True, help='the H&M data: articles.csv and images/')
    parser.add_argument('-output_dir', required=True)
    parser.add_argument('-model_type', required=True, choices=MODEL_TYPES, help='sets the format of the targets')
    parser.add_argument('-seed', type=int, required=True)
    parser.add_argument('-valid_size', type=float, default=5000,
                        help='articles in the validation set; a fraction of all articles if below 1')
    parser.add_argument('-test_size', type=float, default=5000, help='articles in the test set; likewise')
    parser.add_argument('-sp_vocab_size', type=int, default=4000, help='pieces of the SentencePiece model')
    parser.add_argument('-distinct_unk', action='store_true', help='build_vocab.py -distinct_unk')
    parser.add_argument('-missing_values', default='',
                        help='comma-separated values of missing attributes, e.g. Unknown,Undefined,undefined')
    parser.add_argument('-exclude_ids', default='', help='comma-separated ids of articles to leave out')
    opt = parser.parse_args()
    if not 0 <= opt.seed < 2 ** 32:
        parser.error('-seed must be between 0 and 2**32 - 1')
    compat.seed_python_hash(opt.seed)
    compat.seed_everything(opt.seed)
    prepare(opt.hm_dir, opt.output_dir, opt.model_type, opt.seed, opt.valid_size, opt.test_size,
            opt.sp_vocab_size, opt.distinct_unk, opt.missing_values, opt.exclude_ids)


if __name__ == '__main__':
    main()
