import argparse
import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', default=True, action="store_true", help='whether to save results during inference')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1112, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

def inference(_args, model, input_file, outputFile):

    model.eval()
    metric_list = 0.0

    metric_i = test_single_volume(model, 
                                    classes=_args.num_classes,
                                    inputfile=input_file,
                                    outputFile=outputFile,
                                    patch_size=[_args.img_size, _args.img_size])
    return "Testing Finished!"


def runMain(inputFile, outputFile, modelname):

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # name the same snapshot defined in train script!

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cpu()

    snapshot = BASE_DIR+'\\SegmentCalcDir\\model\\' + modelname


    net.load_state_dict(torch.load(snapshot, map_location='cpu'))

    logging.info(str(args))

    inference(args, net, inputFile, outputFile)

