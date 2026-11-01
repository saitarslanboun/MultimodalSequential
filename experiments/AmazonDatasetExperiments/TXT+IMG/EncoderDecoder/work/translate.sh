python ../model/translate.py 			\
	-image_dir /gpfs/hpc/home/hasan90/PAPER1/BaselineAmazonCatModel/dataset	\
	-model models/model.7.chkpt		\
	-img ../dataset/valid.img		\
	-src ../dataset/valid.enc.tit		\
	-vocab ../dataset/data.pt		\
	-batch_size 32
	
	
	
