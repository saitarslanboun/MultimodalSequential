from collections import Counter, OrderedDict

import json
import sys

vocab = json.load(open("train.lbl.json"))
vocab = {v:k for k, v in vocab.iteritems()}

ind = int(sys.argv[1])

l = []
for a in range(1, len(vocab)):
	line = vocab[a]
	atts = line.split()
	if ind == -1:
		val = atts[ind+1]
	else:
		val = atts[ind] + " - " + atts[ind+1]
	l.append(val)	

counts = Counter(l)
sum = 0
for a in range(len(counts)):
	sum += counts[counts.keys()[a]]	

pcounts = dict()
for a in range(len(counts)):
	pcounts[counts.keys()[a]] = float(counts[counts.keys()[a]]) / float(sum)

table = dict()
for a in range(len(pcounts)):
	wval = pcounts.keys()[a]
	val, tval = wval.split(" - ")
	if not val in table.keys():
		table[val] = dict()
	table[val][tval] = pcounts[wval]

kys = table.keys()
target = open("result.txt", "wb")
for a in range(len(kys)):
	skys = table[kys[a]].keys()
	kl = []
	vl = []
	for b in range(len(skys)):
		kl.append(skys[b])
		vl.append(table[kys[a]][skys[b]])
	svl = sorted(range(len(vl)), key=lambda k: vl[k])
	svl.reverse()
	kl = [kl[val] for val in svl]
	vl = [vl[val] for val in svl]
	for b in range(len(kl)):	
		line = kys[a] + " - " + kl[b] + "\t" + str(vl[b]) + "\n"
		target.write(line.encode("utf-8"))
	line = "\n"
	target.write(line)
target.close()
