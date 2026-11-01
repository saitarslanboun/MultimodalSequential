''' Checks that training on the Kaggle data split with a seed is reproducible, with synthetic data: for every
settings file in experiments/kaggle_seeded/ (or the given ones), run.py train (two epochs of a smaller model)
and translate twice with the same seed, once without and once with worker processes. The prepared data, the
weights, the logs and the predictions must be identical. A run with another seed must give another split.

    python tests/reproducibility_test.py [SETTINGS ...] [-keep DIR] [-no_cuda]

Prints one line per settings file and exits with status 1 if any of them fails.
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

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rnmtplus import compat  # noqa: E402
from smoke_test import KAGGLE_OPTIONS, make_kaggle_data  # noqa: E402

TEXT_FILES = ['{}.{}'.format(split, ext) for split in ('train', 'valid', 'test')
              for ext in ('img', 'tit', 'enc.tit', 'lbl')] + ['sp.model', 'sp.vocab', 'train.enc.tit.json',
                                                             'train.lbl.json', 'split.json']


def read(path, mode='rb'):
    with io.open(path, mode) as f:
        return f.read()


def run(settings, stage, data_dir, work, extra, log):
    command = [sys.executable, os.path.join(REPO, 'run.py'), settings, stage, '-data_dir', data_dir,
               '-work_dir', work] + extra
    log.write('$ ' + ' '.join(command) + '\n')
    log.flush()
    if subprocess.call(command, stdout=log, stderr=subprocess.STDOUT):
        raise RuntimeError('{} failed'.format(stage))


def differences(work_a, work_b):
    ''' What differs between the outputs of two runs. '''
    found = []
    for name in TEXT_FILES:
        if read(os.path.join(work_a, 'data', name)) != read(os.path.join(work_b, 'data', name)):
            found.append('data/' + name)
    data = [compat.torch_load(os.path.join(w, 'data', 'data.pt')) for w in (work_a, work_b)]
    for d in data:
        del d['settings']  # holds the paths of the input files
    if data[0] != data[1]:
        found.append('data/data.pt')
    checkpoints = [compat.torch_load(os.path.join(w, 'models', 'model.best.chkpt'), map_location='cpu')
                   for w in (work_a, work_b)]
    weights = [c['model'] for c in checkpoints]
    if list(weights[0]) != list(weights[1]) or \
            any(not torch.equal(weights[0][k], weights[1][k]) for k in weights[0]):
        found.append('weights')
    if checkpoints[0]['epoch'] != checkpoints[1]['epoch']:
        found.append('best epoch')
    for name in ('log.train.log', 'log.valid.log', 'pred_test.txt'):
        if read(os.path.join(work_a, name)) != read(os.path.join(work_b, name)):
            found.append(name)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('settings', nargs='*', help='default: experiments/kaggle_seeded/*/*.json')
    parser.add_argument('-keep', metavar='DIR', help='keep the data and the outputs in DIR')
    parser.add_argument('-no_cuda', action='store_true')
    opt = parser.parse_args()
    settings = opt.settings or sorted(glob.glob(os.path.join(REPO, 'experiments', 'kaggle_seeded', '*', '*.json')))
    root = opt.keep or tempfile.mkdtemp(prefix='reproducibility_test_')
    data_dir = os.path.join(root, 'kaggle_raw')
    make_kaggle_data(data_dir, random.Random(0))
    cuda = ['-no_cuda'] if opt.no_cuda else []
    train = ['-epoch', '2', '-n_layers', '3', '-batch_size', '4', '-save_mode', 'best'] + KAGGLE_OPTIONS + cuda
    other_seed = ['-seed', '2']  # KAGGLE_OPTIONS has -seed 1
    failures = 0
    for path in settings:
        name = os.path.relpath(path, os.path.join(REPO, 'experiments')).replace('.json', '')
        work = lambda run_name: os.path.join(root, 'work', name, run_name)
        start = time.time()
        os.makedirs(work(''))
        with open(os.path.join(root, 'work', name, 'log'), 'w') as log:
            try:
                for run_name, workers in (('a', '0'), ('b', '2')):
                    run(path, 'train', data_dir, work(run_name), train + ['-num_workers', workers], log)
                    run(path, 'translate', data_dir, work(run_name), ['-batch_size', '4', '-num_workers', workers]
                        + cuda, log)
                run(path, 'train', data_dir, work('c'), train + other_seed + ['-epoch', '1'], log)
                found = differences(work('a'), work('b'))
                if read(os.path.join(work('a'), 'data', 'test.img')) == read(os.path.join(work('c'), 'data', 'test.img')):
                    found.append('the test set of another seed is the same')
                status = 'identical' if not found else 'DIFFERENT: ' + ', '.join(found)
            except RuntimeError as e:
                status = 'FAILED: {} (see {})'.format(e, log.name)
        failures += status != 'identical'
        print('%-34s %-10s %5.1f s' % (name, status, time.time() - start))
        sys.stdout.flush()
    if not opt.keep:
        shutil.rmtree(root)
    print('%d of %d failed' % (failures, len(settings)) if failures else 'all %d passed' % len(settings))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
