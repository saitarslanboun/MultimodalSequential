import codecs

MCpred = codecs.open("seen_MC.pred", encoding="utf-8").readlines()
MTpred = codecs.open("seen_MT.pred", encoding="utf-8").readlines()
Spred = codecs.open("seen_S.pred", encoding="utf-8").readlines()
Opred = codecs.open("seen.lbl", encoding="utf-8").readlines()

err_MC = 0
err_MT = 0
err_S = 0
for a in range(len(MCpred)):
	if Opred[a] != MCpred[a]:
		err_MC += 1
	if Opred[a] != MTpred[a]:
                err_MT += 1
	if Opred[a] != Spred[a]:
                err_S += 1

err_MC = 100.0 * float(err_MC) / float(len(MCpred))
err_MT = 100.0 * float(err_MT) / float(len(MCpred))
err_S = 100.0 * float(err_S) / float(len(MCpred))

print ("%.2f" % err_MC)
print ("%.2f" % err_MT)
print ("%.2f" % err_S)
