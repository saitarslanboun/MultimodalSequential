import codecs

MCpred = codecs.open("unseen_MC.pred", encoding="utf-8").readlines()
MLpred = codecs.open("unseen_ML.pred", encoding="utf-8").readlines()
MTpred = codecs.open("unseen_MT.pred", encoding="utf-8").readlines()
AMTpred = codecs.open("unseen_AMT.pred", encoding="utf-8").readlines()
SAMTpred = codecs.open("unseen_SAMT.pred", encoding="utf-8").readlines()
Spred = codecs.open("unseen_S.pred", encoding="utf-8").readlines()
Opred = codecs.open("unseen.lbl", encoding="utf-8").readlines()

err_MC = 0
err_ML = 0
err_MT = 0
err_AMT = 0
err_SAMT = 0
err_S = 0
for a in range(len(MCpred)):
	if Opred[a] != MCpred[a]:
		err_MC += 1
	if Opred[a] != MLpred[a]:
		err_ML += 1
	if Opred[a] != MTpred[a]:
                err_MT += 1
	if Opred[a] != AMTpred[a]:
                err_AMT += 1
	if Opred[a] != SAMTpred[a]:
                err_SAMT += 1
	if Opred[a] != Spred[a]:
                err_S += 1

err_MC = 100.0 * float(err_MC) / float(len(MCpred))
err_ML = 100.0 * float(err_ML) / float(len(MLpred))
err_MT = 100.0 * float(err_MT) / float(len(MCpred))
err_AMT = 100.0 * float(err_AMT) / float(len(MCpred))
err_SAMT = 100.0 * float(err_SAMT) / float(len(MCpred))
err_S = 100.0 * float(err_S) / float(len(MCpred))

print ("%.2f" % err_MC)
print ("%.2f" % err_ML)
print ("%.2f" % err_MT)
print ("%.2f" % err_AMT)
print ("%.2f" % err_SAMT)
print ("%.2f" % err_S)
