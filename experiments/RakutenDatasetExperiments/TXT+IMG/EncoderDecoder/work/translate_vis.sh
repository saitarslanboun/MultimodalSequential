python ../model/translate.py									\
	-image_dir /gpfs/hpc/home/hasan90/PAPER1/Tensor2TensorCatModelProduction/dataset/rimages\
	-model models/model.7.chkpt								\
	-img ../dataset/vis.img								\
	-src ../dataset/vis.tit								\
	-vocab ../dataset/data.pt								\
	-batch_size 1										\
	-output pred_vis.txt
	
	
	
