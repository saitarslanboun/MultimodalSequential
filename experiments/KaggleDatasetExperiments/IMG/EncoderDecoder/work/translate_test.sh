python ../model/translate.py 			\
	-image_dir /gpfs/hpc/home/hasan90/PAPER1/KaggleDatasetExperiments/MultiClass/ndataset/fashion-dataset/images		\
	-model models/model.24.chkpt		\
	-img ../dataset/test.img		\
	-vocab ../dataset/data.pt		\
	-batch_size 32				\
	-output pred_test.txt
	
	
	
