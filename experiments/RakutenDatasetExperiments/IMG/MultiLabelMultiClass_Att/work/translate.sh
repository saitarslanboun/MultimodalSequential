python ../model/translate.py 								\
	-image_dir /gpfs/hpc/home/hasan90/PAPER1/RakutenDatasetExperiments/images	\
	-model models/model.11.chkpt							\
	-img ../dataset/valid.img							\
	-vocab ../dataset/data.pt							\
	-batch_size 32									\
	-output pred.txt
	
	
	
