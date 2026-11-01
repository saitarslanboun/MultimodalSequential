''' Statistics of the attribute chains.

    python analysis/label_stats.py frequency DATA/train.lbl [-column N] [-output graph.png]
        the number of products per chain (or per value of attribute N), by rank, as a plot
    python analysis/label_stats.py pairs WORK/train.lbl.json -position N [-output result.txt]
        over the chains of the multiclass vocabulary: the share of each pair of values of attributes N and N+1
    python analysis/label_stats.py transitions WORK/train.lbl.json -position N [-output result.txt]
        over the chains of the multiclass vocabulary, with <START> before the first attribute: how often
        each value of attribute N-1 (N = 0: <START>) is followed by each value of attribute N
    python analysis/label_stats.py followers DATA/train.lbl -prefix VALUE,VALUE,...
        the values that follow the given first attributes in the training chains
    python analysis/label_stats.py max_length WORK/etrain.lbl.json
        the largest number of attributes of a chain in the multiclass vocabulary (Amazon)

The vocabularies are those of build_vocab.py -mode chains (<UNK> excluded).
'''

import argparse
import io
import json
import os
import sys
from collections import Counter, OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import read_lines  # noqa: E402


def chains(vocab_file):
    ''' The chains of a multiclass vocabulary, in the order of their class index, without <UNK> (0). '''
    with io.open(vocab_file, encoding='utf-8') as f:
        vocab = dict((v, k) for k, v in json.load(f).items())
    return [vocab[a] for a in range(1, len(vocab))]


def frequency(opt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines = read_lines(opt.file)
    if opt.column is not None:
        lines = [line.strip().split()[opt.column] for line in lines]
    counts = [count for _, count in Counter(lines).most_common()]
    plt.plot(list(range(len(counts))), counts)
    plt.xlabel('IDs')
    plt.ylabel('Counts')
    plt.savefig(opt.output or 'graph.png')


def pairs(opt):
    ind = opt.position
    counts = Counter(atts[ind] + " - " + atts[ind + 1] for atts in (c.split() for c in chains(opt.file)))
    total = sum(counts.values())
    table = OrderedDict()
    for pair, count in counts.items():
        val, tval = pair.split(" - ")
        table.setdefault(val, OrderedDict())[tval] = float(count) / float(total)
    with open(opt.output or 'result.txt', 'wb') as target:
        for val, values in table.items():
            for tval, share in sorted(values.items(), key=lambda kv: kv[1], reverse=True):
                target.write((val + " - " + tval + "\t" + str(share) + "\n").encode("utf-8"))
            target.write(b"\n")


def transitions(opt):
    ind = opt.position
    uni_sums = Counter()
    rel_sums = OrderedDict()
    for chain in chains(opt.file):
        atts = ["<START>"] + chain.split()
        uni_sums[atts[ind]] += 1
        rel_sums.setdefault(atts[ind], Counter())[atts[ind + 1]] += 1
    with open(opt.output or 'result.txt', 'wb') as target:
        for mval, followers in rel_sums.items():
            for sval, count in followers.items():
                share = float(count) / float(uni_sums[mval])
                target.write((mval + " -> " + sval + " : " + str((count, share)) + "\n").encode("utf-8"))
            target.write(b"\n")


def followers(opt):
    name = opt.prefix.split(",")
    values = set()
    for line in read_lines(opt.file):
        atts = line.strip().split()
        if atts[:len(name)] == name and len(atts) > len(name):
            values.add(atts[len(name)])
    for value in sorted(values):
        print(value)


def max_length(opt):
    print(max(len(chain.split()) for chain in chains(opt.file)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('statistic', choices=('frequency', 'pairs', 'transitions', 'followers', 'max_length'))
    parser.add_argument('file')
    parser.add_argument('-column', type=int, help='frequency: count the values of this attribute (from 0)')
    parser.add_argument('-position', type=int, default=0, help='pairs, transitions: the attribute (from 0)')
    parser.add_argument('-prefix', help='followers: the first attribute values, separated by commas')
    parser.add_argument('-output')
    opt = parser.parse_args()
    globals()[opt.statistic](opt)


if __name__ == '__main__':
    main()
