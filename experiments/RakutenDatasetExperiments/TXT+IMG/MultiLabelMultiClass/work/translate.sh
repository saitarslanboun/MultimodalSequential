python ../model/translate.py 			\
	-image_dir /gpfs/hpc/home/hasan90/PAPER1/Tensor2TensorCatModelProduction/dataset/rimages		\
	-model models/model.3.chkpt		\
	-img ../dataset/valid.img		\
	-src ../dataset/valid.tit		\
	-vocab ../dataset/data.pt		\
	-batch_size 32
	
	
	
