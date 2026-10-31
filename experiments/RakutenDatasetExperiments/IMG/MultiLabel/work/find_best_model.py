import codecs

f = codecs.open("log.valid.log", encoding="utf-8").readlines()

vals = []
for a in range(1, 32):
	vals.append(float(f[a].strip().split()[1]))

print (vals.index(min(vals)))
