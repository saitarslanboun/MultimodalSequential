''' Error rates of predicted attribute chains (validate.py and test.py of the original experiments).

Rakuten and Kaggle: the error rate of every attribute and of the full chain; a prediction with a single token
(the "None" of the sequential model) counts as wrong for every attribute. Amazon (-chains): a product can
have several chains, listed in the gold file separated by " ||| "; a prediction is correct if it is one of
them. The accuracy of the paper is 100 minus the error rate.
'''

import argparse

from preprocess import read_lines


def attribute_errors(gold, pred, n_attributes):
    ''' The number of errors of every attribute and of the full chain. '''
    errors = [0] * n_attributes
    errors_all = 0
    for a in range(len(gold)):
        olabels = gold[a].strip().split()
        plabels = pred[a].strip().split()
        if len(plabels) == 1:
            errors = [e + 1 for e in errors]
            errors_all += 1
            continue
        for i in range(n_attributes):
            if (olabels[i] if i < len(olabels) else None) != (plabels[i] if i < len(plabels) else None):
                errors[i] += 1
        if olabels != plabels:
            errors_all += 1
    return errors, errors_all


def chain_errors(gold, pred):
    ''' The number of predictions that are none of the chains of their product. '''
    return sum(1 for a in range(len(gold)) if pred[a].strip() not in gold[a].strip().split(" ||| "))


def main():
    parser = argparse.ArgumentParser(description='evaluate.py')
    parser.add_argument('-gold', required=True, help='the attribute chains (valid.lbl, test.lbl; Amazon: '
                                                     'valid.cat, test.cat)')
    parser.add_argument('-pred', required=True, help='the predictions of translate.py')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-n_attributes', type=int, help='the number of attributes (Rakuten 7, Kaggle 8)')
    group.add_argument('-chains', action='store_true', help='Amazon: several gold chains per product')
    opt = parser.parse_args()

    orig = read_lines(opt.gold)
    f = read_lines(opt.pred)
    if len(f) < len(orig):
        parser.error('{} has {} lines, {} has {}'.format(opt.pred, len(f), opt.gold, len(orig)))
    n = len(orig)
    if opt.chains:
        print("Error rate: %" + str((100.0 * float(chain_errors(orig, f))) / n))
        return
    errors, errors_all = attribute_errors(orig, f, opt.n_attributes)
    for i, err in enumerate(errors):
        print("Error rate %d: %%%s" % (i + 1, str((100.0 * float(err)) / n)))
    err_rate_all = (100.0 * float(errors_all)) / n
    print("Error rate all: %" + str(err_rate_all))
    print("Total error: %" + str(err_rate_all))


if __name__ == '__main__':
    main()
