import codecs

test_img = codecs.open("../dataset/test.img", encoding="utf-8").readlines()
test_tit = codecs.open("../dataset/test.tit", encoding="utf-8").readlines()
test_lbl = codecs.open("../dataset/test.cat", encoding="utf-8").readlines()
pred_test_S_lbl = codecs.open("pred_test.txt", encoding="utf-8").readlines()
pred_test_MC_lbl = codecs.open("../../MultiClass/work/pred_test.txt", encoding="utf-8").readlines()
#pred_test_MT_lbl = codecs.open("../../MultiLabelMultiClass/work/pred_test.txt", encoding="utf-8").readlines()
train_lbl = codecs.open("../dataset/etrain.lbl", encoding="utf-8").readlines()

target_img = open("unseen.img", "wb")
target_tit = open("unseen.tit", "wb")
target_lbl = open("unseen.lbl", "wb")
target_S_pred = open("unseen_S.pred", "wb")
target_MC_pred = open("unseen_MC.pred", "wb")
#target_MT_pred = open("unseen_MT.pred", "wb")
for a in range(len(test_lbl)):
	print str(a+1) + " : " + str(len(test_lbl))
	test_lbls = test_lbl[a].strip().split(" ||| ")
	intersection = [i for i in test_lbls if i+"\n" in train_lbl]
	if len(intersection) == 0:
		target_img.write(test_img[a].encode("utf-8"))
		target_tit.write(test_tit[a].encode("utf-8"))
		target_lbl.write(test_lbl[a].encode("utf-8"))
		target_S_pred.write(pred_test_S_lbl[a].encode("utf-8"))
		target_MC_pred.write(pred_test_MC_lbl[a].encode("utf-8"))
		#target_MT_pred.write(pred_test_MT_lbl[a].encode("utf-8"))
target_img.close()
target_tit.close()
target_lbl.close()
target_S_pred.close()
target_MC_pred.close()
#target_MT_pred.close()
