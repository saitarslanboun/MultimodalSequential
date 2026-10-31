python ../model/translate.py		\
	-model models/model.5.chkpt	\
	-src ../dataset/valid.tit	\
	-vocab ../dataset/data.pt	\
	-batch_size 32			\
	-output pred.txt
	
	
	
