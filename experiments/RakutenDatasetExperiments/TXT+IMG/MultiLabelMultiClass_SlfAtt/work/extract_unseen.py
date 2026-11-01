import codecs

test_img = codecs.open("../dataset/test.img", encoding="utf-8").readlines()
test_tit = codecs.open("../dataset/test.tit", encoding="utf-8").readlines()
test_lbl = codecs.open("../dataset/test.lbl", encoding="utf-8").readlines()
pred_test_lbl = codecs.open("pred_test.txt", encoding="utf-8").readlines()
train_lbl = codecs.open("../dataset/train.lbl", encoding="utf-8").readlines()

target_img = open("unseen.img", "wb")
target_tit = open("unseen.tit", "wb")
target_lbl = open("unseen.lbl", "wb")
target_pred = open("unseen.pred", "wb")
for a in range(len(test_lbl)):
	if test_lbl[a] not in train_lbl:
		target_img.write(test_img[a].encode("utf-8"))
		target_tit.write(test_tit[a].encode("utf-8"))
		target_lbl.write(test_lbl[a].encode("utf-8"))
		target_pred.write(str(a+1) + " " + pred_test_lbl[a].encode("utf-8"))
target_img.close()
target_tit.close()
target_lbl.close()
target_pred.close()
