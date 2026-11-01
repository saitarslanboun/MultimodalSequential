''' Error rates of predicted attribute chains (validate.py and test.py of the original experiments).

Rakuten and Kaggle: the error rate of every attribute and of the full chain; a prediction with a single token
(the "None" of the sequential model) counts as wrong for every attribute. Amazon (-chains): a product can
have several chains, listed in the gold file separated by " ||| "; a prediction is correct if it is one of
them. The accuracy of the paper is 100 minus the error rate.

With -missing_values (the Kaggle data: NA,None), a gold value among them marks a missing attribute, which is
not scored: the error rate of an attribute is over the products where it is known, and a chain is correct if
all its known attributes are.
'''

import argparse

from preprocess import read_lines


def known_attributes(chain, n_attributes, missing=()):
    ''' Whether each attribute of a gold chain is known (not one of the missing values). '''
    values = chain.strip().split()
    return [not (i < len(values) and values[i] in missing) for i in range(n_attributes)]


def attribute_errors(gold, pred, n_attributes, missing=()):
    ''' The number of errors of every attribute and of the full chain; the attributes whose gold value is in
    missing are not scored. '''
    errors = [0] * n_attributes
    errors_all = 0
    for a in range(len(gold)):
        olabels = gold[a].strip().split()
        plabels = pred[a].strip().split()
        known = known_attributes(gold[a], n_attributes, missing)
        if len(plabels) == 1:
            errors = [e + int(k) for e, k in zip(errors, known)]
            errors_all += 1
            continue
        for i in range(n_attributes):
            if known[i] and (olabels[i] if i < len(olabels) else None) != (plabels[i] if i < len(plabels) else None):
                errors[i] += 1
        if missing:
            wrong = len(olabels) != len(plabels) or any(
                o != p for o, p in zip(olabels, plabels) if o not in missing)
        else:
            wrong = olabels != plabels
        if wrong:
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
    parser.add_argument('-missing_values', default='', metavar='VALUES',
                        help='comma-separated gold values that mark a missing attribute, e.g. NA,None: not scored')
    opt = parser.parse_args()
    missing = frozenset(v for v in opt.missing_values.split(',') if v)

    orig = read_lines(opt.gold)
    f = read_lines(opt.pred)
    if len(f) < len(orig):
        parser.error('{} has {} lines, {} has {}'.format(opt.pred, len(f), opt.gold, len(orig)))
    n = len(orig)
    if opt.chains:
        print("Error rate: %" + str((100.0 * float(chain_errors(orig, f))) / n))
        return
    errors, errors_all = attribute_errors(orig, f, opt.n_attributes, missing)
    known = [n] * opt.n_attributes
    if missing:
        known = [sum(column) for column in zip(*(known_attributes(chain, opt.n_attributes, missing) for chain in orig))]
    for i, err in enumerate(errors):
        print("Error rate %d: %%%s" % (i + 1, str((100.0 * float(err)) / known[i])))
    err_rate_all = (100.0 * float(errors_all)) / n
    print("Error rate all: %" + str(err_rate_all))
    print("Total error: %" + str(err_rate_all))


if __name__ == '__main__':
    main()
