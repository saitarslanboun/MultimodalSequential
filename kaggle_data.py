''' The Kaggle Fashion Product Images data (styles.csv and images/), split at random with a seed.

train.py -kaggle_dir DIR -seed N calls prepare(), which
- reads the products of styles.csv. As in the paper's data files, the title is the productDisplayName and the
  attribute chain holds the values of gender, masterCategory, subCategory, articleType, baseColour, season,
  year and usage, with the spaces in a value replaced by "_" and "None" for an empty value;
- shuffles the products with numpy.random.RandomState(N), whose numbers are the same in every NumPy version:
  the first -test_size products are the test set, the next -valid_size the validation set, the rest the
  training set;
- trains a SentencePiece model on the training titles, with the settings of the paper's model (unigram, 4,000
  pieces, all characters), and encodes all titles with it;
- builds the vocabularies of the training set (build_vocab.py) and data.pt (preprocess.py).

With missing_values (NA,None), these attribute values mark a missing attribute: they are not part of the target
vocabularies, and data.pt has -1 for them, which the losses ignore. exclude_ids leaves products out of the data
before the split, e.g. 28319, a test record ("RTV_LoadTest") with wrong attributes.

It writes the data directory of the experiment:
  {train,valid,test}.img       the image file names (<id>.jpg), one product per line
  {train,valid,test}.tit       the titles
  {train,valid,test}.enc.tit   the titles as SentencePiece pieces
  {train,valid,test}.lbl       the attribute chains
  sp.model, sp.vocab           the SentencePiece model
  train.enc.tit.json, train.lbl.json   the vocabularies
  data.pt                      the input of train.py and translate.py
  split.json                   the settings: seed, sizes, target format, checksum of styles.csv
The same settings give the same files. To prepare the data without training:

    python kaggle_data.py -kaggle_dir DIR -model_type TYPE -seed N -output_dir OUT
'''

import argparse
import hashlib
import io
import json
import os
from collections import OrderedDict, namedtuple

import numpy as np

import build_vocab
import preprocess
from rnmtplus import compat
from rnmtplus.Models import MODEL_TYPES

ATTRIBUTES = ('gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage')
HEADER = ('id',) + ATTRIBUTES + ('productDisplayName',)
SPLITS = ('train', 'valid', 'test')
TGT_VOCAB_MODES = {'sequence': 'tokens', 'chain': 'chains', 'attributes': 'attributes'}
# The SentencePiece settings of the paper's model, except the number of threads: one thread, so that the
# same titles always give the same model.
SENTENCEPIECE_SETTINGS = dict(model_type='unigram', character_coverage=1.0, input_sentence_size=0,
                              normalization_rule_name='nmt_nfkc', num_threads=1)

Product = namedtuple('Product', ['id', 'image', 'title', 'chain'])


def read_products(kaggle_dir):
    ''' The products of styles.csv, in the order of the file. The titles are not quoted there: a title with
    commas spans several fields, which are joined again. '''
    with io.open(os.path.join(kaggle_dir, 'styles.csv'), encoding='utf-8') as f:
        lines = f.read().splitlines()
    if tuple(lines[0].split(',')) != HEADER:
        raise ValueError('unexpected header in styles.csv: ' + lines[0])
    n_values = len(ATTRIBUTES)
    products = []
    for number, line in enumerate(lines[1:], 2):
        fields = line.split(',')
        if len(fields) < len(HEADER):
            raise ValueError('line {} of styles.csv has {} fields'.format(number, len(fields)))
        values = [value.strip() for value in fields[1:1 + n_values]]
        chain = ' '.join(value.replace(' ', '_') if value else 'None' for value in values)
        products.append(Product(fields[0], fields[0] + '.jpg', ','.join(fields[1 + n_values:]), chain))
    return products


def split_sizes(valid_size, test_size, n):
    ''' The sizes of the validation and test sets: a number of products, or a fraction of n if below 1. '''
    sizes = [int(round(size * n)) if size < 1 else int(size) for size in (valid_size, test_size)]
    if min(sizes) < 1 or sum(sizes) >= n:
        raise ValueError('the validation and test sets need at least one product each, and together fewer '
                         'than the {} products'.format(n))
    return sizes


def split_indices(n, seed, valid_size, test_size):
    ''' The product indices of the three sets, from a permutation of range(n) drawn with
    numpy.random.RandomState(seed): the test set is its first test_size indices, the validation set the next
    valid_size, the training set the rest. '''
    order = np.random.RandomState(seed).permutation(n).tolist()
    return OrderedDict([('train', order[test_size + valid_size:]),
                        ('valid', order[test_size:test_size + valid_size]),
                        ('test', order[:test_size])])


def train_sentencepiece(data_dir, vocab_size, seed):
    ''' Trains a SentencePiece model on data_dir/train.tit (data_dir/sp.model and sp.vocab) and returns it. The
    model records the names of its input and output files, so it is trained in data_dir with the names
    train.tit and sp, as the paper's model was: the same titles then give the same file in any directory. '''
    import sentencepiece as spm
    if hasattr(spm, 'set_random_generator_seed'):
        spm.set_random_generator_seed(seed)
    cwd = os.getcwd()
    os.chdir(data_dir)
    try:
        spm.SentencePieceTrainer.Train(input='train.tit', model_prefix='sp', vocab_size=vocab_size,
                                       **SENTENCEPIECE_SETTINGS)
    finally:
        os.chdir(cwd)
    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(data_dir, 'sp.model'))
    return sp


def write_lines(path, lines):
    with io.open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + u'\n')


def file_md5(path):
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            md5.update(block)
    return md5.hexdigest()


def prepare(kaggle_dir, output_dir, model_type, seed, valid_size=5000, test_size=5000, sp_vocab_size=4000,
            distinct_unk=False, missing_values='', exclude_ids=''):
    ''' Writes the data directory of an experiment (see above) and returns the path of its data.pt. A directory
    that was prepared with other settings is not overwritten. distinct_unk: build_vocab.py -distinct_unk.
    missing_values, exclude_ids: comma-separated values and product ids (see above). '''
    if not 0 <= seed < 2 ** 32:
        raise ValueError('the seed must be between 0 and 2**32 - 1')
    checksum = ('styles_csv_md5', file_md5(os.path.join(kaggle_dir, 'styles.csv')))
    return prepare_products(read_products(kaggle_dir), checksum, kaggle_dir, output_dir, model_type, seed,
                            valid_size, test_size, sp_vocab_size, distinct_unk, missing_values, exclude_ids)


def prepare_products(products, checksum, source_dir, output_dir, model_type, seed, valid_size=5000,
                     test_size=5000, sp_vocab_size=4000, distinct_unk=False, missing_values='', exclude_ids=''):
    ''' The steps of prepare() for the products (Product) of a dataset in source_dir; hm_data.py prepares the H&M
    data with them. checksum: the key and the checksum of the dataset's file, the first entry of split.json. '''
    if not 0 <= seed < 2 ** 32:
        raise ValueError('the seed must be between 0 and 2**32 - 1')
    missing = [v for v in missing_values.split(',') if v]
    excluded = [i for i in exclude_ids.split(',') if i]
    products = [p for p in products if p.id not in excluded]
    n_valid, n_test = split_sizes(valid_size, test_size, len(products))
    tgt_format = preprocess.TGT_FORMATS[model_type]
    settings = OrderedDict([checksum, ('seed', seed),
                            ('valid_size', n_valid), ('test_size', n_test), ('sp_vocab_size', sp_vocab_size),
                            ('tgt_format', tgt_format), ('distinct_unk', bool(distinct_unk)),
                            ('missing_values', missing), ('exclude_ids', excluded)])
    split_file = os.path.join(output_dir, 'split.json')
    if os.path.exists(split_file):
        with io.open(split_file, encoding='utf-8') as f:
            previous = json.load(f)
        different = [key for key in settings if previous.get(key) != settings[key]]
        if different:
            raise SystemExit('{} holds data prepared with another {}; give another directory'.format(
                output_dir, ', '.join(different)))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    path = lambda name: os.path.join(output_dir, name)
    print('[Info] Splitting the {} products of {} with seed {}: {} training, {} validation, {} test'.format(
        len(products), source_dir, seed, len(products) - n_valid - n_test, n_valid, n_test))

    splits = split_indices(len(products), seed, n_valid, n_test)
    for name, indices in splits.items():
        items = [products[i] for i in indices]
        write_lines(path(name + '.img'), [p.image for p in items])
        write_lines(path(name + '.tit'), [p.title for p in items])
        write_lines(path(name + '.lbl'), [p.chain for p in items])
    sp = train_sentencepiece(output_dir, sp_vocab_size, seed)
    for name, indices in splits.items():
        write_lines(path(name + '.enc.tit'), [' '.join(sp.EncodeAsPieces(products[i].title)) for i in indices])

    unk = ['-distinct_unk'] if distinct_unk else []
    build_vocab.main(['-mode', 'tokens', '-output_dir', output_dir, path('train.enc.tit')] + unk)
    build_vocab.main(['-mode', TGT_VOCAB_MODES[tgt_format], '-output_dir', output_dir,
                      '-missing_values', ','.join(missing), path('train.lbl')] + unk)
    args = ['-model_type', model_type, '-save_data', path('data.pt'), '-missing_values', ','.join(missing),
            '-src_vocab', path('train.enc.tit.json'), '-tgt_vocab', path('train.lbl.json')]
    for name in ('train', 'valid'):
        args += ['-{}_img'.format(name), path(name + '.img'), '-{}_src'.format(name), path(name + '.enc.tit'),
                 '-{}_tgt'.format(name), path(name + '.lbl')]
    preprocess.main(args)

    import sentencepiece
    settings['products'] = OrderedDict((name, len(indices)) for name, indices in splits.items())
    settings['source'] = os.path.abspath(source_dir)
    settings['sentencepiece_version'] = getattr(sentencepiece, '__version__', 'unknown')
    with io.open(split_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(settings, indent=2, ensure_ascii=False) + u'\n')
    return path('data.pt')


def main():
    parser = argparse.ArgumentParser(description='Splits the Kaggle data with a seed and prepares the data '
                                                 'directory of an experiment, as train.py -kaggle_dir does.')
    parser.add_argument('-kaggle_dir', required=True, help='the Kaggle data: styles.csv and images/')
    parser.add_argument('-output_dir', required=True)
    parser.add_argument('-model_type', required=True, choices=MODEL_TYPES, help='sets the format of the targets')
    parser.add_argument('-seed', type=int, required=True)
    parser.add_argument('-valid_size', type=float, default=5000,
                        help='products in the validation set; a fraction of all products if below 1')
    parser.add_argument('-test_size', type=float, default=5000, help='products in the test set; likewise')
    parser.add_argument('-sp_vocab_size', type=int, default=4000, help='pieces of the SentencePiece model')
    parser.add_argument('-distinct_unk', action='store_true', help='build_vocab.py -distinct_unk')
    parser.add_argument('-missing_values', default='', help='comma-separated values of missing attributes, e.g. NA,None')
    parser.add_argument('-exclude_ids', default='', help='comma-separated ids of products to leave out')
    opt = parser.parse_args()
    if not 0 <= opt.seed < 2 ** 32:
        parser.error('-seed must be between 0 and 2**32 - 1')
    compat.seed_python_hash(opt.seed)
    compat.seed_everything(opt.seed)
    prepare(opt.kaggle_dir, opt.output_dir, opt.model_type, opt.seed, opt.valid_size, opt.test_size,
            opt.sp_vocab_size, opt.distinct_unk, opt.missing_values, opt.exclude_ids)


if __name__ == '__main__':
    main()
