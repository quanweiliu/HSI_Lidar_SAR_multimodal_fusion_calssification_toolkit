import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description='MDL')
    # general
    parser.add_argument('--model_name', type=str, default='MDL')
    parser.add_argument('--path-config', type=str, default='')
    parser.add_argument('--print-config', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # datasets
    parser.add_argument('--dataset_name', type=str, default='HOSDSAR')   # IndianPines
    parser.add_argument('--path_data', type=str, default="/home/leo/MMF/OSD")
    parser.add_argument('--data_info_start', default=0, type=int)
    parser.add_argument('--print-data-info', action='store_true', default=False)

    parser.add_argument('--classes', type=int, default=15)
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--expansion_factor', action='store_true', default=False)
    parser.add_argument('--pca', action='store_true', default=False)
    parser.add_argument('--components', type=int, default=32)

    parser.add_argument('--split_type', type=str, default='disjoint')
    parser.add_argument('--train_num', type=int, default=1000)
    parser.add_argument('--val_num', type=int, default=1000)
    parser.add_argument('--train_ratio', type=float, default=0.1)
    parser.add_argument('--val_ratio', type=int, default=0)
    parser.add_argument('--show_gt', type=str, default=False)
    parser.add_argument('--remove_zero_labels', action='store_true', default=True)
    parser.add_argument('--distillation', type=int, default=1, help='distillation') #是否加蒸馏

    
    
    parser.add_argument('--SD',  nargs='+', default=["PaviaU", "Salinas", "Dioni", 'MelasChasma', 'GaleCrater', 'CopratesChasma'], 
                        help='SingleModality data')
    parser.add_argument('--MD',  nargs='+', default=["Houston_2013", "Houston_2018", "Augsburg", "Berlin"], 
                        help='MultiModality data')
    
    # model
    parser.add_argument('--SSISO',  nargs='+', default=["vit", 'cnn', 'mamba'], 
                        help='singlesacle singlemodality input and singleoutput')
    parser.add_argument('--SSISO2',  nargs='+', default=['ViTDGCN', 'FDGC', "SSFTTnet", \
                                                         "morphFormer", "DBCTNet"], 
                        help='singlesacle singlemodality input and singleoutput')
    

    
    
    parser.add_argument('--SMISO',  nargs='+', default=['S2ENet', 'FusAtNet', 'CrossHL', \
                                                        'HCTNet', 'SHNet', \
                                                        "MDL_M", "MDL_L", "EndNet"], 
                        help='singlesacle multimodelity input and singleoutput')
    parser.add_argument('--SMIMO',  nargs='+', default=["MS2CANet"], 
                        help='singlesacle multimodelity input and multioutput')
    parser.add_argument('--SMIMO2',  nargs='+', default=["MDL_E_D"], 
                        help='singlesacle multimodelity input and multioutput')
    parser.add_argument('--SMIMO3',  nargs='+', default=["MDL_C"], 
                        help='singlesacle multimodelity input and multioutput')
    parser.add_argument('--MMISO',  nargs='+', default=['DSHFNet'], \
                        help='mutlisacle multimodelity input and singleoutput')
    parser.add_argument('--MMIMO',  nargs='+', default=["MIViT", ], \
                        help='mutlisacle multimodelity input and multiouput')   

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
    parser.add_argument('--lb_smooth', type=float, default=0.01)
    parser.add_argument('--num_nodes', type=int, default=1, help="# 1,2,3  Multiples of classes")
    parser.add_argument('--feature_dim', default=126, type=int, help='Feature dim for latent vector')

    parser.add_argument('--plot-loss-curve', action='store_true', default=False)
    parser.add_argument('--show-results', action='store_true', default=False)


    # log
    parser.add_argument('--log_interval', default=1, type=int)
    parser.add_argument('--resume', default='', type=str, help='resume weights')


    parser.add_argument('--save-results', action='store_true', default=True)
    parser.add_argument('--path_head', default='/home/leo/Oil_Spill_Detection/SHNet', 
                        type=str, help='path to cache (default: none)')
    parser.add_argument('--path_result_dir', type=str, default='')

    '''
    args = parser.parse_args()  # running in command line
    '''
    args = parser.parse_args('')  # running in ipynb
    return args

