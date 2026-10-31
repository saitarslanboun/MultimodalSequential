python preprocess.py					\
	-train_src ../dataset/etrain.tit		\
	-train_tgt ../dataset/etrain.lbl		\
	-valid_src ../dataset/evalid.tit		\
	-valid_tgt ../dataset/evalid.lbl		\
	-src_vocab ../dataset/etrain.tit.json		\
	-tgt_vocab ../dataset/etrain.lbl.json		\
	-save_data ../dataset/data.pt
