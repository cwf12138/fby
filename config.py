import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # in/out
    parser.add_argument('--outf', default='./experiments',
                        help='trained model will be saved at here')
    parser.add_argument('--save', default='resnet18',
                        help='save name of experiment in args.outf directory')

    # data
    parser.add_argument('--train_path',
                        default='./dataset/train.txt')
    parser.add_argument('--test_path',
                        default='./dataset/test.txt')

    # model parameters
    parser.add_argument('--in_channels', default=3,type=int,
                        help='input channels of model')
    parser.add_argument('--classes', default=4,type=int, 
                        help='output channels of model')

    # training
    parser.add_argument('--N_epochs', default=50, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=16,
                        type=int, help='batch size')
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                    help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                    help='initial learning rate')
    parser.add_argument('--val_on_test', default=True, type=bool,
                        help='Validation on testset')

    # for pre_trained checkpoint
    parser.add_argument('--start_epoch', default=1, 
                        help='Start epoch')
    parser.add_argument('--pre_trained', default=None,
                        help='(path of trained _model) load trained model to continue train')

    # hardware setting
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use GPU calculating')

    args = parser.parse_args()

    return args
