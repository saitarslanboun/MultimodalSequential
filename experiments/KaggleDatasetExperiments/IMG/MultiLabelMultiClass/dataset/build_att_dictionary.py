from collections import OrderedDict
import fileinput
import sys

import numpy
import json


def main():
    for filename in sys.argv[1:]:
        print('Processing', filename)
        word_freqs = OrderedDict()
        with open(filename, 'r') as f:
            for line in f:
                words_in = line.strip().split()
                if len(word_freqs.keys()) == 0:
                    for a in range(len(words_in)):
                        word_freqs[a] = OrderedDict()
                for a in range(len(words_in)):
                    if words_in[a] not in word_freqs[a]:
                        word_freqs[a][words_in[a]] = 0
                    word_freqs[a][words_in[a]] += 1
        words = []
        freqs = []
        for a in range(len(word_freqs)):
            words.append(list(word_freqs[a].keys()))
            freqs.append(list(word_freqs[a].values()))

        sorted_idx = []
        sorted_words = []
        for a in range(len(freqs)):
            sorted_idx.append(numpy.argsort(freqs[a]))
            sorted_words.append([words[a][ii] for ii in sorted_idx[a][::-1]])

        worddict = OrderedDict()
        for a in range(len(word_freqs)):
            worddict[a] = OrderedDict()
            worddict[a]['<UNK>'] = 0
            # FIXME We shouldn't assume <EOS>, <GO>, and <UNK> aren't BPE subwords.
            for ii, ww in enumerate(sorted_words[a]):
                worddict[a][ww] = ii+1

        # The JSON RFC requires that JSON text be represented using either
        # UTF-8, UTF-16, or UTF-32, with UTF-8 being recommended.
        # We use UTF-8 regardless of the user's locale settings.
        with open('%s.json'%filename, 'w') as f:
            json.dump(worddict, f, indent=2, ensure_ascii=False)

        print('Done')

if __name__ == '__main__':
	main()
