''' The epoch with the lowest validation loss (or the highest accuracy) in a log of train.py.

    python analysis/best_epoch.py WORK/log.valid.log [-metric accuracy|chain_accuracy] [-max_epoch N]
'''

import argparse
import io


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('log')
    parser.add_argument('-metric', choices=('loss', 'accuracy', 'chain_accuracy'), default='loss')
    parser.add_argument('-max_epoch', type=int, help='only look at the epochs up to this one')
    opt = parser.parse_args()

    with io.open(opt.log, encoding='utf-8') as f:
        header = f.readline().strip().split(',')
        rows = [dict(zip(header, line.strip().split(','))) for line in f if line.strip()]
    if opt.metric not in header:
        parser.error('{} has no {} column'.format(opt.log, opt.metric))
    rows = [r for r in rows if opt.max_epoch is None or int(r['epoch']) <= opt.max_epoch]
    best = (min if opt.metric == 'loss' else max)(rows, key=lambda r: float(r[opt.metric]))
    print('epoch {}: {} {}'.format(best['epoch'], opt.metric, best[opt.metric].strip()))


if __name__ == '__main__':
    main()
