r"""config"""
import argparse

def parse_opts():
    r"""arguments"""
    parser = argparse.ArgumentParser(description='Dense Cross-Query-and-Support Attention Weighted Mask Aggregation for Few-Shot Segmentation')

    # common
    parser.add_argument('--datapath', type=str, default=r'../../data/FSS-1000')
    parser.add_argument('--pascal_datapath', type=str, default=r'../../data/VOCdevkit')
    parser.add_argument('--coco_datapath', type=str, default=r'../../data/COCO')
    parser.add_argument('--fss_datapath', type=str, default=r'../../data/FSS-1000')
    parser.add_argument('--city_datapath', type=str, default=r'../../data/cityscapes')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'fss', 'coco', 'ade20k', 'cityscapes'])
    parser.add_argument('--test_dataset', type=str, default='Animal', choices = ['Animal', 'Artificial_Luna_Landscape', 'ClinicDB', 'Crack_Detection', 'Eyeballs', 'Magnetic_tile_surface', 'Leaf_Disease', 'Aerial'])
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--backbone', type=str, default='swin-l', choices=['resnet50', 'resnet101', 'swin-b', 'swin-l', 'vgg'])
    parser.add_argument('--feature_extractor_path', type=str, default='./swin_base_patch4_window12_384_22kto1k.pth')
    parser.add_argument('--logpath', type=str, default='./logs')
    parser.add_argument('--model_name', type=str, default='')

    # for train
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--nepoch', type=int, default=1) # 原来为1000
    parser.add_argument('--distributed', type=bool, default='False')
    parser.add_argument('--resume', type=str, default='')
    # for test
    parser.add_argument('--load', type=str, default='./pascal-5i/swin_fold0.pt')
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--vispath', type=str, default='./vis')
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--test_num', type=int, default=1000)
    parser.add_argument('--test_datapath', type=str, default=r'../../data/Dataset')
    parser.add_argument('--test_epoch', type=int, default=2)
    parser.add_argument('--vote', action='store_true')
    parser.add_argument('--average', action='store_true')
    parser.add_argument('--neptune', action='store_true')

    # for model
    parser.add_argument('--num_queries', type=int, default=100)
    parser.add_argument('--mask_dim', type=int, default=256)
    parser.add_argument('--conv_dim', type=int, default=256)
    parser.add_argument('--add_mask', action='store_true')
    parser.add_argument('--dec_layers', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--enforce_input_project', type=bool, default=False)


    args = parser.parse_args()
    return args

