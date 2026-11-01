from collections import Counter

import sys
import codecs
import matplotlib.pyplot as plt

f = codecs.open(sys.argv[1], encoding="utf-8").readlines()

most_common = Counter(f).most_common()

vals = []
IDs = []
for a in range(len(most_common)):
	vals.append(most_common[a][1])
	IDs.append(a)

plt.plot(IDs, vals)
plt.xlabel('IDs')
plt.ylabel('Counts')
plt.savefig("graph.png")
