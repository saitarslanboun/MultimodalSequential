python ../model/train.py												\
	-data ../dataset/data.pt 											\
	-save_model models 												\
	-save_mode all 													\
	-batch_size 32 													\
	-log log 													\
	-image_dir /gpfs/hpc/home/hasan90/PAPER1/KaggleDatasetExperiments/images	\
	-n_warmup_steps 4000												\
	-d_model 1024

