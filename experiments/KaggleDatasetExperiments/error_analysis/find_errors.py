import codecs

txt = codecs.open("pred_txt.txt", encoding="utf-8").readlines()
mul = codecs.open("pred_mul.txt", encoding="utf-8").readlines()
lbl = codecs.open("test.lbl", encoding="utf-8").readlines()
titles = codecs.open("test.enc.tit", encoding="utf-8").readlines()
images = codecs.open("test.img", encoding="utf-8").readlines()

itarget = open("ind", "wb")
ttarget = open("txt", "wb")
mtarget = open("mul", "wb")
tittarget = open("title", "wb")
imgtarget = open("image", "wb")
cntr = 0
for a in range(len(txt)):
	if txt[a] == lbl[a] and mul[a] != lbl[a]:
		itarget.write((str(a+1)+"\n").encode("utf-8"))
		tittarget.write(titles[a].encode("utf-8"))
		imgtarget.write(images[a].encode("utf-8"))
		ttarget.write(txt[a].encode("utf-8"))
		mtarget.write(mul[a].encode("utf-8"))
		cntr += 1
	if cntr == 50:
		break
itarget.close()
ttarget.close()
mtarget.close()
tittarget.close()
imgtarget.close()
