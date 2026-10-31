python ../model/translate.py 			\
	-model models/model.12.chkpt		\
	-src ../dataset/test.enc.tit		\
	-vocab ../dataset/data.pt		\
	-batch_size 32				\
	-output pred_test.txt
	
	
	
