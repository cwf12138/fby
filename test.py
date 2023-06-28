import torch.backends.cudnn as cudnn
import torch.optim as optim
import sys, time
from os.path import join
import torch
import torch.nn as nn
from config import parse_args
from dataset import *
from utils.function import *
import warnings
from utils.common import *
warnings.filterwarnings('ignore')

from models.pretrained.resnet18 import *

def main():

    args = parse_args()
    save_path = join(args.outf, args.save)

    save_args(args,save_path)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    cudnn.benchmark = True
    print('The computing device used is: ','GPU' if device.type=='cuda' else 'CPU')

    # 定义模型
    net = Resnet18(classes=args.classes).to(device)

    checkpoint = torch.load('experiments/resnet18/dental_fluorosis_resnet18.pth',map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    
    test_loader = load_test(args)
        
    # train stage
    acc = test(test_loader, net, None, device, args)

    print(round(acc, 4))


if __name__ == '__main__':
    main()
