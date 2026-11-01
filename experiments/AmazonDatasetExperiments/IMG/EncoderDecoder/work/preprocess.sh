python preprocess.py				\
	-train_img ../dataset/etrain.img	\
	-train_tgt ../dataset/etrain.lbl	\
	-valid_img ../dataset/evalid.img	\
	-valid_tgt ../dataset/evalid.lbl	\
	-tgt_vocab ../dataset/etrain.lbl.json	\
	-save_data ../dataset/data.pt
