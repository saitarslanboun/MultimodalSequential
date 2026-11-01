''' Builds the vocabularies of an experiment: for every input file FILE, writes FILE.json.

Modes:
- tokens: a token vocabulary, for titles and for the attribute chains of the sequential model; the
  indices 0-3 are <BLANK>, <BOS>, <EOS> and <UNK>, then the tokens by decreasing frequency. As in the
  original code, the most frequent token also gets index 3, the index of <UNK>.
- chains: the class vocabulary of the multiclass model: <UNK> (0), then the attribute chains (whole
  lines) by decreasing frequency.
- attributes: one vocabulary per attribute (position in the chain) for the multilabel and multitask
  models, saved under the keys "0", "1", ...: <UNK> (0), then the values by decreasing frequency.
'''

import argparse
import io
import json
import os
from collections import OrderedDict

SMALL_QUICKSORT = 15


def _aheapsort(v, tosort, start, n):
    a = lambda i: start + i - 1  # 1-based heap index -> position in tosort
    for l in range(n >> 1, 0, -1):
        tmp = tosort[a(l)]
        i, j = l, l << 1
        while j <= n:
            if j < n and v[tosort[a(j)]] < v[tosort[a(j + 1)]]:
                j += 1
            if v[tmp] < v[tosort[a(j)]]:
                tosort[a(i)] = tosort[a(j)]
                i, j = j, j + j
            else:
                break
        tosort[a(i)] = tmp
    while n > 1:
        tmp = tosort[a(n)]
        tosort[a(n)] = tosort[a(1)]
        n -= 1
        i, j = 1, 2
        while j <= n:
            if j < n and v[tosort[a(j)]] < v[tosort[a(j + 1)]]:
                j += 1
            if v[tmp] < v[tosort[a(j)]]:
                tosort[a(i)] = tosort[a(j)]
                i, j = j, j + j
            else:
                break
        tosort[a(i)] = tmp


def argsort(v):
    ''' numpy.argsort(v) as NumPy 1.x computes it (introsort). The vocabularies of the experiments were built
    with it; NumPy 2 orders equal frequencies differently, which would give the words other indices. '''
    num = len(v)
    tosort = list(range(num))
    if num < 2:
        return tosort
    stack = []
    pl, pr = 0, num - 1
    cdepth = (num.bit_length() - 1) * 2
    while True:
        if cdepth < 0:
            _aheapsort(v, tosort, pl, pr - pl + 1)
        else:
            while pr - pl > SMALL_QUICKSORT:
                pm = pl + ((pr - pl) >> 1)
                if v[tosort[pm]] < v[tosort[pl]]:
                    tosort[pm], tosort[pl] = tosort[pl], tosort[pm]
                if v[tosort[pr]] < v[tosort[pm]]:
                    tosort[pr], tosort[pm] = tosort[pm], tosort[pr]
                if v[tosort[pm]] < v[tosort[pl]]:
                    tosort[pm], tosort[pl] = tosort[pl], tosort[pm]
                vp = v[tosort[pm]]
                pi, pj = pl, pr - 1
                tosort[pm], tosort[pj] = tosort[pj], tosort[pm]
                while True:
                    pi += 1
                    while v[tosort[pi]] < vp:
                        pi += 1
                    pj -= 1
                    while vp < v[tosort[pj]]:
                        pj -= 1
                    if pi >= pj:
                        break
                    tosort[pi], tosort[pj] = tosort[pj], tosort[pi]
                tosort[pi], tosort[pr - 1] = tosort[pr - 1], tosort[pi]
                cdepth -= 1
                if pi - pl < pr - pi:
                    stack.append((pi + 1, pr, cdepth))
                    pr = pi - 1
                else:
                    stack.append((pl, pi - 1, cdepth))
                    pl = pi + 1
            for pi in range(pl + 1, pr + 1):  # insertion sort
                vi = tosort[pi]
                vp = v[vi]
                pj = pi
                while pj > pl and vp < v[tosort[pj - 1]]:
                    tosort[pj] = tosort[pj - 1]
                    pj -= 1
                tosort[pj] = vi
        if not stack:
            return tosort
        pl, pr, cdepth = stack.pop()


def by_decreasing_frequency(word_freqs):
    words = list(word_freqs.keys())
    freqs = list(word_freqs.values())
    sorted_idx = argsort(freqs)
    return [words[ii] for ii in sorted_idx[::-1]]


def build_token_vocab(lines):
    word_freqs = OrderedDict()
    for line in lines:
        for w in line.strip().split(' '):
            if w not in word_freqs:
                word_freqs[w] = 0
            word_freqs[w] += 1
    worddict = OrderedDict()
    worddict['<BLANK>'] = 0
    worddict['<BOS>'] = 1
    worddict['<EOS>'] = 2
    worddict['<UNK>'] = 3
    for ii, ww in enumerate(by_decreasing_frequency(word_freqs)):
        worddict[ww] = ii + 3
    return worddict


def build_chain_vocab(lines):
    word_freqs = OrderedDict()
    for line in lines:
        w = line.strip()
        if w not in word_freqs:
            word_freqs[w] = 0
        word_freqs[w] += 1
    worddict = OrderedDict()
    worddict['<UNK>'] = 0
    for ii, ww in enumerate(by_decreasing_frequency(word_freqs)):
        worddict[ww] = ii + 1
    return worddict


def build_attribute_vocab(lines):
    word_freqs = OrderedDict()
    for line in lines:
        words_in = line.strip().split()
        if len(word_freqs.keys()) == 0:
            for a in range(len(words_in)):
                word_freqs[a] = OrderedDict()
        for a in range(len(words_in)):
            if words_in[a] not in word_freqs[a]:
                word_freqs[a][words_in[a]] = 0
            word_freqs[a][words_in[a]] += 1
    worddict = OrderedDict()
    for a in range(len(word_freqs)):
        worddict[a] = OrderedDict()
        worddict[a]['<UNK>'] = 0
        for ii, ww in enumerate(by_decreasing_frequency(word_freqs[a])):
            worddict[a][ww] = ii + 1
    return worddict


BUILDERS = {'tokens': build_token_vocab, 'chains': build_chain_vocab, 'attributes': build_attribute_vocab}


def main(argv=None):
    parser = argparse.ArgumentParser(description='build_vocab.py')
    parser.add_argument('-mode', required=True, choices=sorted(BUILDERS))
    parser.add_argument('-output_dir', help='where to write the .json files (default: next to the inputs)')
    parser.add_argument('files', nargs='+')
    opt = parser.parse_args(argv)
    for filename in opt.files:
        print('Processing', filename)
        with io.open(filename, encoding='utf-8') as f:
            worddict = BUILDERS[opt.mode](f)
        output = filename + '.json'
        if opt.output_dir:
            output = os.path.join(opt.output_dir, os.path.basename(output))
        # The JSON RFC requires UTF-8, UTF-16 or UTF-32; UTF-8 regardless of the locale.
        with io.open(output, 'w', encoding='utf-8') as f:
            f.write(json.dumps(worddict, indent=2, ensure_ascii=False))
        print('Done')


if __name__ == '__main__':
    main()
