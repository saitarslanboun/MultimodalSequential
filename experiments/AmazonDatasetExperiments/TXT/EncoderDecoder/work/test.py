import codecs, sys

orig = codecs.open("test.cat", encoding="utf-8").readlines()
f = codecs.open("pred_test.txt", encoding="utf-8").readlines()

err = 0
for a in range(len(orig)):
	olabels = orig[a].strip().split(" ||| ")
	plabel = f[a].strip()

	if plabel not in olabels:
		err += 1

err_rate = (100.0 * float(err)) / 5000.0
print ("Error rate: %" + str(err_rate))
