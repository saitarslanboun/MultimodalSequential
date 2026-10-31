python ../model/translate.py 								\
	-image_dir /gpfs/hpc/home/hasan90/PAPER1/RakutenDatasetExperiments/images	\
	-model models/model.6.chkpt							\
	-img ../dataset/valid.img							\
	-src ../dataset/valid.tit							\
	-vocab ../dataset/data.pt							\
	-batch_size 32									\
	-output pred.txt
	
	
	
