import torch.nn as nn
from torchvision import models

class MyResNet(nn.Module):

    def __init__(self, in_channels=1, out_features = 512):
        super(MyResNet, self).__init__()

        # bring resnet
        self.model = models.resnet18()
        # original definition of the first layer on the renset class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=out_features, bias=True)
        

    def forward(self, x):
        return self.model(x)