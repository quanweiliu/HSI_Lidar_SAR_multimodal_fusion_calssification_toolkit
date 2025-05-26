# from MS2CANet.pymodel import pyCNN

# __all__ = ["pyCNN"]


def get_model_config(args):

	if args.backbone == "cnn" or args.backbone == "vit":
		args.epochs = 100
		args.batch_size = 64
		args.patch_size = 13
		args.randomCrop = 11
		args.pca = True
		args.components = 15
		# args.pca = False
		# args.components = 0
		args.learning_rate = 0.001
		##### original version no step_size and gamma
		args.schedule = True
		args.step_size = 30
		args.gamma = 0.7

	elif args.backbone == "mamba":
		args.epochs = 100
		args.batch_size = 64
		args.patch_size = 12
		args.randomCrop = 8
		args.pca = True
		args.components = 15    
		# args.pca = False
		# args.components = 0
		args.learning_rate = 0.001
		##### original version no step_size and gamma
		args.schedule = True
		args.step_size = 30
		args.gamma = 0.7

	elif args.backbone == "ViTDGCN":
		args.data_augmentation = False
		args.randomCrop = None
		args.batch_size = 128
		args.patch_size = 27
		args.pca = True
		args.components = 35
		# args.pca = False
		# args.components = 0
		args.learning_rate = 0.0005
		args.schedule = True
		args.step_size = 30
		args.gamma = 0.5

	elif args.backbone == "FDGC":
		args.data_augmentation = False
		args.randomCrop = None
		args.batch_size = 128
		args.patch_size = 19
		args.pca = True
		args.components = 32
		# args.pca = False
		# args.components = 0
		args.learning_rate = 0.0001
		args.schedule = True
		args.step_size = 30
		args.gamma = 0.5

	elif args.backbone == "SSFTTnet":
		args.data_augmentation = False
		args.randomCrop = None
		args.batch_size = 64
		args.patch_size = 13
		args.pca = True
		args.components = 30
		# args.pca = False      # NO
		# args.components = 0
		args.learning_rate = 0.001
		args.schedule = False
		# args.step_size = 30
		# args.gamma = 0.5

	elif args.backbone == "morphFormer":
		args.epochs = 500
		args.data_augmentation = False
		args.randomCrop = None
		args.batch_size = 64
		args.patch_size = 11
		args.pca = False
		args.components = 0
		args.learning_rate = 5e-4
		args.weight_decay = 5e-3
		args.FM = 16
		args.testSizeNumber = 100
		args.schedule = True
		args.step_size = 50
		args.gamma = 0.9

		# args.epochs = 100
		# args.data_augmentation = True
		# # args.randomCrop = 11
		# args.batch_size = 256
		# args.patch_size = 13
		# args.pca = False
		# args.components = 0
		# args.learning_rate = 1e-3
		# args.weight_decay = 1e-4
		# args.FM = 16
		# args.HSIOnly = True
		# args.testSizeNumber = 100
		# args.schedule = True
		# args.step_size = 30
		# args.gamma = 0.7

	elif args.backbone == "DBCTNet":
		# args.epochs = 300   # PU
		args.epochs = 500   # Houston
		# args.epochs = 600
		args.batch_size = 128
		args.patch_size = 9
		args.pca = False
		args.components = 0    
		# args.learning_rate = 0.001
		args.learning_rate = 0.003 # Houston
		args.weight_decay = 0.001
		# args.gammaF = 0.4  # PU
		args.gammaF = 1.7  # Houston

		##### original version no step_size and gamma
		args.schedule = True
		args.step_size = 30
		args.gamma = 0.7

	elif args.backbone == "MDL_M" or args.backbone == "MDL_L" or \
		args.backbone == "MDL_E_D" or args.backbone == "MDL_C":
		args.epochs = 150
		args.patch_size = 7
		args.batch_size = 64
		args.pca = False
		args.components = 0
		args.learning_rate = 0.001
		##### original version no step_size and gamma
		args.schedule = True
		args.step_size = 30
		args.gamma = 0.5

	elif args.backbone == "FusAtNet":
		args.epochs = 1000
		args.patch_size = 11
		args.batch_size = 64
		args.pca = False
		args.components = 0
		args.learning_rate = 0.000005
		##### original version no step_size and gamma
		args.schedule = False
		# args.step_size = 30
		# args.gamma = 0.5

	elif  args.backbone == "CrossHL":
		args.epochs = 200
		args.patch_size = 11
		args.batch_size = 64
		args.pca = False
		args.components = 0
		args.learning_rate = 0.0005
		args.schedule = True
		args.step_size = 50
		args.gamma = 0.9

	elif args.backbone == "MS2CANet":
		args.epochs = 100
		args.patch_size = 11
		args.batch_size = 64
		args.learning_rate = 0.001
		args.pca = True
		args.components = 20    # Houston
		# args.pca = False
		# args.components = 0 
		##### original version no step_size and gamma
		args.schedule = False
		# args.step_size = 30
		# args.gamma = 0.5

	elif args.backbone == "S2ENet":
		args.epochs = 128
		args.batch_size = 64
		args.patch_size = 7
		args.pca = True
		args.components = 20    # Houston
		# args.pca = False
		# args.components = 0
		args.learning_rate = 0.001
		args.schedule = True

	elif args.backbone == "HCTNet":
		args.epochs = 100
		args.batch_size = 64
		args.patch_size = 11
		args.pca = True
		args.components = 20    # Houston
		# # args.components = 30    # Trento
		# args.pca = False      # NO
		# args.components = 0
		args.learning_rate = 0.001
		##### original version no step_size and gamma
		args.schedule = False
		# args.step_size = 30
		# args.gamma = 0.5

	elif args.backbone == "SHNet":
		args.epochs = 100
		args.batch_size = 128
		args.patch_size = 7
		args.pca = False
		args.components = 0
		args.factors = 4
		# args.factors = 8
		args.learning_rate = 0.001
		args.weight_decay = 0.001
		args.lb_smooth = 0.01
		args.schedule = True
		args.step_size = 30
		args.gamma = 0.5

	elif args.backbone == "DSHFNet":
		args.epochs = 500
		args.batch_size = 64
		args.patch_size = 6
		args.pca = False
		args.components = 0
		args.learning_rate = 5e-4
		args.weight_decay = 0
		args.schedule = True
		args.step_size = args.epochs // 10
		args.gamma = 0.9

	elif args.backbone == "MIViT":
		args.epochs = 500
		args.patch_size = 8
		args.batch_size = 64
		args.distillation = 1
		args.pca = False
		args.components = 0
		args.learning_rate = 1e-4
		args.weight_decay = 0.001
		args.schedule = True
		args.step_size = args.epochs // 10
		args.gamma = 0.9
		args.fusion = 'TTOA'
		args.pred_flag = 'o_fuse'
		
	else:
		raise ValueError(f"Unsupported backbone: {args.backbone}")