''' Full-chain error rates on the test products whose attribute chain occurs in the training set ("seen") and
on the others ("unseen"): Table 5 (Kaggle) and Section 5.1 (Rakuten, Amazon).

    python analysis/seen_unseen.py -train_lbl DATA/train.lbl -test_lbl DATA/test.lbl \
        -pred MC=WORK_MC/pred_test.txt -pred S=WORK_S/pred_test.txt ...

Each -pred gives a name and the test predictions of a model (translate.py). As in the original scripts, the
lines are compared exactly. Amazon (-chains): -test_lbl is test.cat (the chains of a product separated by
" ||| ") and -train_lbl etrain.lbl; a product is unseen if none of its chains occurs in training, and a
prediction is correct if it is one of its chains.

-write DIR also writes the subsets, as the original scripts did: DIR/seen.img, seen.tit, seen.lbl and
seen_<name>.pred for every model, and the same for unseen (with -test_img and -test_tit; -numbered starts
the prediction lines with the line number of the product in the test set).
'''

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import read_lines  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-train_lbl', required=True)
    parser.add_argument('-test_lbl', required=True)
    parser.add_argument('-pred', action='append', required=True, metavar='NAME=FILE')
    parser.add_argument('-chains', action='store_true', help='Amazon: several chains per test product')
    parser.add_argument('-write', metavar='DIR')
    parser.add_argument('-test_img')
    parser.add_argument('-test_tit')
    parser.add_argument('-numbered', action='store_true')
    opt = parser.parse_args()

    train_lbl = set(read_lines(opt.train_lbl))
    test_lbl = read_lines(opt.test_lbl)
    preds = []
    for spec in opt.pred:
        name, _, path = spec.partition('=')
        if not path:
            parser.error('-pred NAME=FILE')
        preds.append((name, read_lines(path)))

    if opt.chains:
        seen = [any(c + "\n" in train_lbl for c in line.strip().split(" ||| ")) for line in test_lbl]
        correct = lambda gold, pred: pred.strip() in gold.strip().split(" ||| ")
    else:
        seen = [line in train_lbl for line in test_lbl]
        correct = lambda gold, pred: gold == pred

    for subset, in_subset in (('seen', True), ('unseen', False)):
        items = [a for a in range(len(test_lbl)) if seen[a] == in_subset]
        print('{}: {} products'.format(subset, len(items)))
        for name, pred in preds:
            errors = sum(1 for a in items if not correct(test_lbl[a], pred[a]))
            print('  {}: {}'.format(name, '%.2f' % (100.0 * float(errors) / float(len(items))) if items else '-'))
        if opt.write:
            write_subset(opt, subset, items, test_lbl, preds)


def write_subset(opt, subset, items, test_lbl, preds):
    if not os.path.isdir(opt.write):
        os.makedirs(opt.write)
    files = [(subset + '.lbl', test_lbl)]
    if opt.test_img:
        files.append((subset + '.img', read_lines(opt.test_img)))
    if opt.test_tit:
        files.append((subset + '.tit', read_lines(opt.test_tit)))
    for name, lines in files:
        with open(os.path.join(opt.write, name), 'wb') as f:
            for a in items:
                f.write(lines[a].encode('utf-8'))
    for name, pred in preds:
        with open(os.path.join(opt.write, '{}_{}.pred'.format(subset, name)), 'wb') as f:
            for a in items:
                f.write(((str(a + 1) + ' ' if opt.numbered else '') + pred[a]).encode('utf-8'))


if __name__ == '__main__':
    main()
