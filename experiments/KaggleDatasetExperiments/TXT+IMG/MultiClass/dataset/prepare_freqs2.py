from collections import Counter, OrderedDict

import json
import sys

vocab = json.load(open("train.lbl.json"))
vocab = {v:k for k, v in vocab.iteritems()}

ind = int(sys.argv[1])

uni_sums = dict()
for a in range(1, len(vocab)):
	line = vocab[a]
	atts = line.split()
	atts = ["<START>"] + atts
	val = atts[ind]
	if val not in uni_sums.keys():
		uni_sums[val] = 0
	uni_sums[val] += 1

rel_sums = dict()
for a in range(1, len(vocab)):
	line = vocab[a]
	atts = line.split()
	atts = ["<START>"] + atts
	mval = atts[ind]
	if mval not in rel_sums.keys():
		rel_sums[mval] = dict()
	sbal = atts[ind+1]
	if sbal not in rel_sums[mval].keys():
		rel_sums[mval][sbal] = 0
	rel_sums[mval][sbal] += 1

target = open("result.txt", "wb")
kys = rel_sums.keys()
for a in range(len(kys)):
	skys = rel_sums[kys[a]].keys()
	for b in range(len(skys)):
		inte = rel_sums[kys[a]][skys[b]]
		perc = float(inte) / float(uni_sums[kys[a]])
		line = kys[a] + " -> " + skys[b] + " : " + str((inte, perc)) + "\n"
		target.write(line.encode("utf-8"))
	target.write("\n")
target.close()
