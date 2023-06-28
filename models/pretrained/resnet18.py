from torchvision.models import resnet18
import torch
from torch import nn

class Resnet18(nn.Module):
    def __init__(self, classes):
        super(Resnet18, self).__init__()
        self.backbone = resnet18(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, classes)

    def forward(self, x):
        x = self.backbone(x)
        return x


if __name__ == '__main__':
    net = Resnet18(classes=4).cuda()

    inputs = torch.randn(4, 3, 448, 560).cuda()

    out1 = net(inputs)

    print(out1.shape)