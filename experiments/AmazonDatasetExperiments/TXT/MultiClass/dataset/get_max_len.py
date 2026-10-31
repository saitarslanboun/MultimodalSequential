import json

vocab = json.load(open("etrain.lbl.json"))

max_len = 0
for a in range(len(vocab.keys())):
	length = len(vocab.keys()[a].split())
	if length == 8:
		print vocab.keys()[a], a
		exit()
	max_len = max(max_len, length)

print (max_len)
