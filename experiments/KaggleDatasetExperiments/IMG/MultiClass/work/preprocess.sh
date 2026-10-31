python preprocess.py					\
	-train_img ../dataset/train.img			\
	-train_tgt ../dataset/train.lbl			\
	-valid_img ../dataset/valid.img			\
	-valid_tgt ../dataset/valid.lbl			\
	-tgt_vocab ../dataset/train.lbl.json		\
	-save_data ../dataset/data.pt
