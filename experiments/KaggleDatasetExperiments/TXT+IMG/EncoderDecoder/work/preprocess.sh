python preprocess.py					\
	-train_img ../dataset/train.img			\
	-train_src ../dataset/train.enc.tit		\
	-train_tgt ../dataset/train.lbl			\
	-valid_img ../dataset/valid.img			\
	-valid_src ../dataset/valid.enc.tit		\
	-valid_tgt ../dataset/valid.lbl			\
	-src_vocab ../dataset/train.enc.tit.json	\
	-tgt_vocab ../dataset/train.lbl.json		\
	-save_data ../dataset/data.pt
