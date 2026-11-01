python ../model/translate.py 							\
	-image_dir /gpfs/hpc/home/hasan90/PAPER1/KaggleDatasetExperiments/images\
	-model models/model.9.chkpt						\
	-img ../dataset/test.img						\
	-src ../dataset/test.enc.tit						\
	-vocab ../dataset/data.pt						\
	-batch_size 32								\
	-output pred_test.txt
	
	
	
