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
'''

import argparse
import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ('vocab', 'preprocess', 'train', 'translate', 'evaluate')
PREDICTIONS = {'valid': 'pred.txt', 'test': 'pred_test.txt'}


def script(name):
    return os.path.join(HERE, name)


def flags(options):
    ''' Command-line flags for the options of a settings file: true is a flag without value. '''
    args = []
    for key, value in options.items():
        if value is True:
            args.append('-' + key)
        elif value is not False and value is not None:
            args += ['-' + key, str(value)]
    return args


def kaggle_commands(settings, opt):
    ''' The commands of the stage for the Kaggle data split with -seed (experiments/kaggle_seeded): train
    prepares the data in WORK/data. '''
    work = lambda name: os.path.join(opt.work_dir, name)
    prepared = lambda name: os.path.join(opt.work_dir, 'data', name)
    model_type, modality = settings['model_type'], settings['modality']
    image_dir = opt.image_dir or os.path.join(opt.data_dir, 'images')
    if opt.stage in ('vocab', 'preprocess'):
        sys.exit('{} has no {} stage: train splits the Kaggle data with -seed and prepares it in '
                 'WORK/data'.format(opt.settings, opt.stage))
    if opt.stage == 'train':
        args = ['-model_type', model_type, '-modality', modality, '-kaggle_dir', opt.data_dir,
                '-kaggle_data_dir', work('data'), '-save_model', work('models'), '-log', work('log')]
        if 'img' in modality:
            args += ['-image_dir', image_dir]
        return [[script('train.py')] + args + flags(settings['kaggle']) + flags(settings['train'])]
    if opt.stage == 'translate':
        args = ['-model_type', model_type, '-model', opt.checkpoint or work(os.path.join('models', 'model.best.chkpt')),
                '-vocab', prepared('data.pt'), '-output', work(PREDICTIONS[opt.split])]
        if 'img' in modality:
            args += ['-img', prepared(opt.split + '.img'), '-image_dir', image_dir]
        if 'txt' in modality:
            args += ['-src', prepared(opt.split + '.enc.tit')]
        return [[script('translate.py')] + args + flags(settings['translate'])]
    args = ['-gold', prepared(opt.split + '.lbl'), '-pred', work(PREDICTIONS[opt.split]),
            '-n_attributes', str(settings['evaluate']['n_attributes'])]
    return [[script('evaluate.py')] + args]


def commands(settings, opt):
    ''' The commands (argument lists) of the stage. '''
    if 'kaggle' in settings:
        return kaggle_commands(settings, opt)
    data = lambda name: os.path.join(opt.data_dir, name)
    work = lambda name: os.path.join(opt.work_dir, name)
    model_type = settings['model_type']
    images = 'img' in settings['modality']
    if images and opt.stage in ('train', 'translate') and not opt.image_dir:
        sys.exit('{} uses images: give -image_dir'.format(opt.settings))

    if opt.stage == 'vocab':
        modes = {'src_vocab': 'tokens',
                 'tgt_vocab': {'sequential': 'tokens', 'multiclass': 'chains'}.get(model_type, 'attributes')}
        return [[script('build_vocab.py'), '-mode', mode, '-output_dir', opt.work_dir,
                 data(settings['preprocess'][key][:-len('.json')])]
                for key, mode in sorted(modes.items()) if key in settings['preprocess']]

    if opt.stage == 'preprocess':
        args = ['-model_type', model_type, '-save_data', work('data.pt')]
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
        if not opt.checkpoint:
            sys.exit('give the checkpoint to translate with: -checkpoint WORK/models/model.<epoch>.chkpt '
                     '(the paper used {})'.format(settings['paper_checkpoint']))
        args = ['-model_type', model_type, '-model', opt.checkpoint, '-vocab', work('data.pt'),
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
    return [[script('evaluate.py')] + args]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument('settings', help='experiments/<dataset>/<input>/<model>.json')
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('-data_dir', required=True,
                        help='the data files of the dataset (experiments/kaggle_seeded: the Kaggle data)')
    parser.add_argument('-image_dir', help='the images (image models; experiments/kaggle_seeded: default '
                                           'DATA/images)')
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
