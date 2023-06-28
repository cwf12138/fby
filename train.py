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
    setpu_seed(42)

    args = parse_args()
    save_path = join(args.outf, args.save)

    save_args(args,save_path)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    cudnn.benchmark = True
    print('The computing device used is: ','GPU' if device.type=='cuda' else 'CPU')

    # 定义模型
    net = Resnet18(classes=args.classes).to(device)

    print("Total number of parameters: " + str(count_parameters(net)))


    if args.pre_trained is not None:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.outf + '%s/latest_model.pth' % args.pre_trained)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']+1

    # 损失函数
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    
    # 优化器
    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    # 学习策略
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_epochs, eta_min=0)
    
    train_loader, val_loader = load_data(args)
    
    if args.val_on_test: 
        print('\033[0;32m===============Validation on Testset!!!===============\033[0m')

    best = {'epoch':0,'accuary':0.5} # Initialize the best epoch and performance(AUC of ROC)

    for epoch in range(args.start_epoch,args.N_epochs+1):
        print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
            (epoch, args.N_epochs,optimizer.state_dict()['param_groups'][0]['lr'], time.asctime()))
        
        # train stage
        train(train_loader, net, criterion1, criterion2, optimizer, device) 
        acc = test(val_loader, net, criterion1, device, args)

        # lr_scheduler.step()

        # Save checkpoint of latest and best model.
        state = {'net': net.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, join(save_path, 'latest_model.pth'))
        
        print('\033[0;33mSaving model!\033[0m')
        torch.save(state, join(save_path, f'model_{epoch}.pth'))

        if acc > best['accuary']:
            best['epoch'] = epoch
            best['accuary'] = acc

        print('Best performance at Epoch: {} | accuary: {}'.format(best['epoch'],best['accuary']))
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
