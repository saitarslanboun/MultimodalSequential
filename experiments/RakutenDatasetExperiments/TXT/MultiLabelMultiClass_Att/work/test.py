import codecs, sys

orig = codecs.open("test.lbl", encoding="utf-8").readlines()
f = codecs.open("pred_test.txt", encoding="utf-8").readlines()

err1 = 0
err2 = 0
err3 = 0
err4 = 0
err5 = 0
err6 = 0
err7 = 0
err_all = 0
for a in range(len(orig)):
	olabels = orig[a].strip().split()
	plabels = f[a].strip().split()
	if len(plabels) == 1:
		err1 += 1
		err2 += 1
		err3 += 1
		err4 += 1
		err5 += 1
		err6 += 1
		err7 += 1
		err_all += 1
		continue

	if olabels[0] != plabels[0]:
		err1 += 1
	if olabels[1] != plabels[1]:
                err2 += 1
	if olabels[2] != plabels[2]:
                err3 += 1
	if olabels[3] != plabels[3]:
                err4 += 1
	if olabels[4] != plabels[4]:
                err5 += 1
	if olabels[5] != plabels[5]:
                err6 += 1
	if olabels[6] != plabels[6]:
                err7 += 1
	if olabels != plabels:
		err_all += 1

err_rate1 = (100.0 * float(err1)) / 5000.0
err_rate2 = (100.0 * float(err2)) / 5000.0
err_rate3 = (100.0 * float(err3)) / 5000.0
err_rate4 = (100.0 * float(err4)) / 5000.0
err_rate5 = (100.0 * float(err5)) / 5000.0
err_rate6 = (100.0 * float(err6)) / 5000.0
err_rate7 = (100.0 * float(err7)) / 5000.0
err_rate_all = (100.0 * float(err_all)) / 5000.0
print ("Error rate 1: %" + str(err_rate1))
print ("Error rate 2: %" + str(err_rate2))
print ("Error rate 3: %" + str(err_rate3))
print ("Error rate 4: %" + str(err_rate4))
print ("Error rate 5: %" + str(err_rate5))
print ("Error rate 6: %" + str(err_rate6))
print ("Error rate 7: %" + str(err_rate7))
print ("Error rate all: %" + str(err_rate_all))
print ("Total error: %" + str(err_rate_all))
