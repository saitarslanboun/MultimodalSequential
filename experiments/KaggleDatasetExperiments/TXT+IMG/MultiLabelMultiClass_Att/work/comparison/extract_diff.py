import codecs

o = codecs.open("test.lbl", encoding="utf-8").readlines()
mc = codecs.open("mc.txt", encoding="utf-8").readlines()
mlmc = codecs.open("mlmc.txt", encoding="utf-8").readlines()
ed = codecs.open("ed.txt", encoding="utf-8").readlines()

targeta = open("diff_mlmc.txt", "wb")
targetb = open("diff_ed.txt", "wb")
targetc = open("diff_mc.txt", "wb")
for a in range(len(mlmc)):
	lo = o[a].strip().split()
	lmc = mc[a].strip().split()
	lmlmc = mlmc[a].strip().split()
	led = ed[a].strip().split()
	if ([lmc[4], lmlmc[4], led[4]] == list(set([lmc[4], lmlmc[4], led[4]]))) and led[4] == lo[4]:
		targeta.write(str(a+1) + "\t" + mlmc[a].encode("utf-8"))
		targetb.write(str(a+1) + "\t" + ed[a].encode("utf-8"))
		targetc.write(str(a+1) + "\t" + mc[a].encode("utf-8"))
targeta.close()
targetb.close()
targetc.close()
