import sys
import codecs

ind = int(sys.argv[1])
name = list(sys.argv[2].split(","))

f = codecs.open("train.lbl", encoding="utf-8").readlines()

attributes = []
for a in range(len(f)):
	atts = f[a].strip().split()
	if atts[:ind+1] == name:
		attributes.append(atts[ind+1])
print set(attributes)
