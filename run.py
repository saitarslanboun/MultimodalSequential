''' Runs a stage of one of the experiments of the paper, with the settings in its file in experiments/.

    python run.py SETTINGS STAGE -data_dir DATA [-image_dir IMAGES] [-work_dir WORK] [stage options]

SETTINGS is a file experiments/<dataset>/<input>/<model>.json. DATA holds the data files of the dataset
(train.img, train.tit, train.lbl, ...), IMAGES the images; everything the experiment creates goes to WORK
(default: the current directory).

Stages:
  vocab       build the vocabularies from the training files in DATA, into WORK (build_vocab.py)
  preprocess  convert the data files into WORK/data.pt (preprocess.py)
  train       train the model: checkpoints in WORK/models, logs in WORK/log.{train,valid}.log (train.py)
  translate   predict the validation set (-split valid: WORK/pred.txt) or the test set (-split test:
              WORK/pred_test.txt) with -checkpoint (translate.py)
  evaluate    the error rates of WORK/pred.txt or WORK/pred_test.txt (evaluate.py)

Other options are passed on to the script of the stage, after those of the settings file, which they
override; for example: python run.py SETTINGS train -data_dir DATA -image_dir IMAGES -save_mode best -patience 3

The settings in experiments/kaggle_seeded/ split the Kaggle data with a seed (kaggle_data.py). DATA is then the
Kaggle data (styles.csv and images/), and there are no vocab and preprocess stages: train splits the data with
-seed and prepares it in WORK/data, which translate and evaluate use.

    python run.py SETTINGS train -data_dir KAGGLE -work_dir WORK -seed 1
    python run.py SETTINGS translate -data_dir KAGGLE -work_dir WORK    (WORK/models/model.best.chkpt)
    python run.py SETTINGS evaluate -data_dir KAGGLE -work_dir WORK

The settings in experiments/hm_seeded/ do the same with the H&M data (hm_data.py): DATA is then the directory of
articles.csv and images/. Those in experiments/fashiongen_seeded/ with the FashionGen data (fashiongen_data.py):
DATA is the export of fashiongen_export.py (products.csv and images/), whose test set is fixed and whose
validation set -seed draws.
'''

import argparse
import io
import json
import os
import subprocess
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ('vocab', 'preprocess', 'train', 'translate', 'evaluate')
PREDICTIONS = {'valid': 'pred.txt', 'test': 'pred_test.txt'}


def script(name):
    return os.path.join(HERE, name)


def flags(options):
    ''' Command-line flags for the options of a settings file: true is a flag without value, a list a
    comma-separated value. '''
    args = []
    for key, value in options.items():
        if value is True:
            args.append('-' + key)
        elif isinstance(value, list):
            args += ['-' + key, ','.join(str(v) for v in value)]
        elif value is not False and value is not None:
            args += ['-' + key, str(value)]
    return args


def missing_flags(settings):
    ''' -missing_values for the settings with "missing_values" (the attribute values of missing attributes). '''
    return flags({'missing_values': settings['missing_values']}) if settings.get('missing_values') else []


def default_checkpoint(settings, opt):
    ''' The checkpoint to translate with: -checkpoint, or the one that -save_mode best or last keeps. '''
    if opt.checkpoint:
        return opt.checkpoint
    save_mode = settings['train'].get('save_mode')
    if save_mode in ('best', 'last'):
        return os.path.join(opt.work_dir, 'models', 'model.{}.chkpt'.format(save_mode))
    return None


def data_files(settings):
    ''' The data files that the settings name: the inputs of preprocess, translate and evaluate. '''
    names = [v for k, v in settings.get('preprocess', {}).items() if k.startswith(('train_', 'valid_'))]
    for split in ('valid', 'test'):
        names += list(settings.get('translate', {}).get(split, {}).values())
        if split in settings.get('evaluate', {}):
            names.append(settings['evaluate'][split])
    return sorted(set(names))


def exclude_products(settings, opt):
    ''' For settings with "exclude_ids" (product ids, e.g. 28319, a test record of the Kaggle data with wrong
    attributes): writes the data files of the settings without the lines of these products to WORK/data, which the
    stages then read instead of DATA. The products of a file <split>.* are those of the image list <split>.img
    (one per line, <id>.jpg). The vocab stage writes the files; the other stages write them if they are missing.
    Returns the directory of the data files. '''
    if not settings.get('exclude_ids'):
        return opt.data_dir
    out = os.path.join(opt.work_dir, 'data')
    record = os.path.join(out, 'excluded.json')
    if opt.dry_run or (opt.stage != 'vocab' and os.path.exists(record)):
        return out
    images = set('{}.jpg'.format(i) for i in settings['exclude_ids'])
    if not os.path.isdir(out):
        os.makedirs(out)
    removed = {}
    for name in data_files(settings):
        split = name.split('.')[0]
        with io.open(os.path.join(opt.data_dir, split + '.img'), encoding='utf-8') as f:
            drop = set(k for k, line in enumerate(f.read().split('\n')) if line.strip() in images)
        with open(os.path.join(opt.data_dir, name), 'rb') as f:
            lines = f.read().split(b'\n')
        trailing = lines[-1] == b''
        lines = lines[:-1] if trailing else lines
        kept = [line for k, line in enumerate(lines) if k not in drop]
        with open(os.path.join(out, name), 'wb') as f:
            f.write(b'\n'.join(kept) + (b'\n' if trailing else b''))
        removed[name] = len(lines) - len(kept)
    with io.open(record, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'exclude_ids': settings['exclude_ids'], 'lines_removed': removed}, indent=2,
                           sort_keys=True) + u'\n')
    print('[Info] {} without the products {}: lines removed {}'.format(out, ', '.join(settings['exclude_ids']), removed))
    return out


# The datasets split with -seed: the key of their settings, the option of train.py, the name, the data
# directory option of train.py.
SPLIT_DATASETS = OrderedDict([('kaggle', ('-kaggle_dir', 'Kaggle', '-kaggle_data_dir')),
                              ('hm', ('-hm_dir', 'H&M', '-split_data_dir')),
                              ('fashiongen', ('-fashiongen_dir', 'FashionGen', '-split_data_dir'))])


def split_commands(settings, opt):
    ''' The commands of the stage for the Kaggle, H&M or FashionGen data split with -seed (experiments/kaggle_seeded,
    experiments/hm_seeded, experiments/fashiongen_seeded): train prepares the data in WORK/data. '''
    key = next(k for k in SPLIT_DATASETS if k in settings)
    option, name, data_option = SPLIT_DATASETS[key]
    work = lambda name: os.path.join(opt.work_dir, name)
    prepared = lambda name: os.path.join(opt.work_dir, 'data', name)
    model_type, modality = settings['model_type'], settings['modality']
    image_dir = opt.image_dir or os.path.join(opt.data_dir, 'images')
    if opt.stage in ('vocab', 'preprocess'):
        sys.exit('{} has no {} stage: train splits the {} data with -seed and prepares it in '
                 'WORK/data'.format(opt.settings, opt.stage, name))
    if opt.stage == 'train':
        args = ['-model_type', model_type, '-modality', modality, option, opt.data_dir,
                data_option, work('data'), '-save_model', work('models'), '-log', work('log')]
        if 'img' in modality:
            args += ['-image_dir', image_dir]
        return [[script('train.py')] + args + missing_flags(settings) +
                flags({'exclude_ids': settings.get('exclude_ids')}) + flags(settings[key]) +
                flags(settings['train'])]
    if opt.stage == 'translate':
        args = ['-model_type', model_type, '-model', default_checkpoint(settings, opt),
                '-vocab', prepared('data.pt'), '-output', work(PREDICTIONS[opt.split])]
        if 'img' in modality:
            args += ['-img', prepared(opt.split + '.img'), '-image_dir', image_dir]
        if 'txt' in modality:
            args += ['-src', prepared(opt.split + '.enc.tit')]
        return [[script('translate.py')] + args + flags(settings['translate'])]
    args = ['-gold', prepared(opt.split + '.lbl'), '-pred', work(PREDICTIONS[opt.split]),
            '-n_attributes', str(settings['evaluate']['n_attributes'])]
    return [[script('evaluate.py')] + args + missing_flags(settings)]


def commands(settings, opt):
    ''' The commands (argument lists) of the stage. '''
    if any(k in settings for k in SPLIT_DATASETS):
        return split_commands(settings, opt)
    data_dir = exclude_products(settings, opt)
    data = lambda name: os.path.join(data_dir, name)
    work = lambda name: os.path.join(opt.work_dir, name)
    model_type = settings['model_type']
    images = 'img' in settings['modality']
    if images and opt.stage in ('train', 'translate') and not opt.image_dir:
        sys.exit('{} uses images: give -image_dir'.format(opt.settings))

    if opt.stage == 'vocab':
        modes = {'src_vocab': 'tokens',
                 'tgt_vocab': {'sequential': 'tokens', 'multiclass': 'chains'}.get(model_type, 'attributes')}
        return [[script('build_vocab.py'), '-mode', mode, '-output_dir', opt.work_dir,
                 data(settings['preprocess'][key][:-len('.json')])] + flags(settings.get('vocab', {})) +
                (missing_flags(settings) if key == 'tgt_vocab' else [])
                for key, mode in sorted(modes.items()) if key in settings['preprocess']]

    if opt.stage == 'preprocess':
        args = ['-model_type', model_type, '-save_data', work('data.pt')] + missing_flags(settings)
        for key, value in settings['preprocess'].items():
            if key in ('src_vocab', 'tgt_vocab'):
                args += ['-' + key, work(value)]
            elif key.startswith(('train_', 'valid_')):
                args += ['-' + key, data(value)]
            else:
                args += flags({key: value})
        return [[script('preprocess.py')] + args]

    if opt.stage == 'train':
        args = ['-model_type', model_type, '-modality', settings['modality'], '-data', work('data.pt'),
                '-save_model', work('models'), '-log', work('log')]
        if images:
            args += ['-image_dir', opt.image_dir]
        return [[script('train.py')] + args + flags(settings['train'])]

    split_files = settings['translate'][opt.split]
    if opt.stage == 'translate':
        checkpoint = default_checkpoint(settings, opt)
        if not checkpoint:
            sys.exit('give the checkpoint to translate with: -checkpoint WORK/models/model.<epoch>.chkpt '
                     '(the paper used {})'.format(settings.get('paper_checkpoint')))
        args = ['-model_type', model_type, '-model', checkpoint, '-vocab', work('data.pt'),
                '-output', work(PREDICTIONS[opt.split])]
        if 'img' in split_files:
            args += ['-img', data(split_files['img']), '-image_dir', opt.image_dir]
        if 'src' in split_files:
            args += ['-src', data(split_files['src'])]
        options = dict((k, v) for k, v in settings['translate'].items() if k not in PREDICTIONS)
        return [[script('translate.py')] + args + flags(options)]

    evaluate = settings['evaluate']
    args = ['-gold', data(evaluate[opt.split]), '-pred', work(PREDICTIONS[opt.split])]
    args += ['-chains'] if evaluate.get('chains') else ['-n_attributes', str(evaluate['n_attributes'])]
    return [[script('evaluate.py')] + args + missing_flags(settings)]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument('settings', help='experiments/<dataset>/<input>/<model>.json')
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('-data_dir', required=True,
                        help='the data files of the dataset (experiments/kaggle_seeded: the Kaggle data; '
                             'experiments/hm_seeded: the H&M data; experiments/fashiongen_seeded: the export of '
                             'fashiongen_export.py)')
    parser.add_argument('-image_dir', help='the images (image models; experiments/kaggle_seeded, hm_seeded and '
                                           'fashiongen_seeded: default DATA/images)')
    parser.add_argument('-work_dir', default='.', help='for data.pt, vocabularies, models, logs and predictions')
    parser.add_argument('-split', choices=sorted(PREDICTIONS), default='test', help='translate, evaluate')
    parser.add_argument('-checkpoint', help='translate: the checkpoint to use')
    parser.add_argument('-dry_run', action='store_true', help='only print the commands')
    opt, extra = parser.parse_known_args()

    with io.open(opt.settings, encoding='utf-8') as f:
        settings = json.load(f)
    if not opt.dry_run and not os.path.isdir(opt.work_dir):
        os.makedirs(opt.work_dir)
    for command in commands(settings, opt):
        command = [sys.executable] + command + extra
        print(' '.join(command))
        sys.stdout.flush()
        if not opt.dry_run:
            subprocess.check_call(command)


if __name__ == '__main__':
    main()
