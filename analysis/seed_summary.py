''' The error rates of runs with different seeds, and their mean and standard deviation over the runs.

    python analysis/seed_summary.py -n_attributes 8 WORK [WORK ...]                       (experiments/kaggle_seeded)
    python analysis/seed_summary.py -n_attributes 7 -gold DATA/test.lbl WORK [WORK ...]   (the paper's splits)
    python analysis/seed_summary.py -chains -gold DATA/test.cat WORK [WORK ...]           (Amazon)

WORK is the work directory of a run (run.py -work_dir): its predictions are WORK/pred_test.txt (WORK/pred.txt
with -split valid). The gold chains are those of -gold, or WORK/data/test.lbl (valid.lbl) for the Kaggle data
split with a seed, where every seed has its own test set. The error rates are those of evaluate.py; the
standard deviation is that of a sample (n - 1).
'''

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import attribute_errors, chain_errors  # noqa: E402
from preprocess import read_lines  # noqa: E402


def mean_std(values):
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, float('nan')
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('work_dirs', nargs='+', metavar='WORK')
    parser.add_argument('-gold', help='the gold chains of all runs (default: WORK/data/<split>.lbl)')
    parser.add_argument('-split', choices=('valid', 'test'), default='test')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-n_attributes', type=int, help='the number of attributes (Rakuten 7, Kaggle 8)')
    group.add_argument('-chains', action='store_true', help='Amazon: several gold chains per product')
    opt = parser.parse_args()

    names = (['chain'] if opt.chains else
             ['attribute %d' % (i + 1) for i in range(opt.n_attributes)] + ['full chain'])
    rates = []
    for work in opt.work_dirs:
        gold = read_lines(opt.gold or os.path.join(work, 'data', opt.split + '.lbl'))
        pred = read_lines(os.path.join(work, 'pred_test.txt' if opt.split == 'test' else 'pred.txt'))
        if len(pred) < len(gold):
            parser.error('{} has fewer predictions than gold chains'.format(work))
        n = float(len(gold))
        if opt.chains:
            rates.append([100 * chain_errors(gold, pred) / n])
        else:
            errors, errors_all = attribute_errors(gold, pred, opt.n_attributes)
            rates.append([100 * e / n for e in errors] + [100 * errors_all / n])

    width = max(len(w) for w in opt.work_dirs + ['mean +- std'])
    print('error rates (%)')
    print(' ' * width + ''.join('%14s' % name for name in names))
    for work, row in zip(opt.work_dirs, rates):
        print(work.ljust(width) + ''.join('%14.2f' % r for r in row))
    summary = [mean_std([row[i] for row in rates]) for i in range(len(names))]
    print('mean +- std'.ljust(width) + ''.join('%14s' % ('%.2f +- %.2f' % ms) for ms in summary))
    print('runs: %d' % len(rates))


if __name__ == '__main__':
    main()
