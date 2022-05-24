import torch.nn as nn
from torchvision import models
import torch
from torch import Tensor

class MyResNet(nn.Module):

    def __init__(self, in_channels=1, out_features = 512, isPredictor = True):
        super(MyResNet, self).__init__()

        # bring resnet
        self.model = models.resnet18()
        # original definition of the first layer on the renset class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=out_features, bias=True)
        if (isPredictor):
            self.model.predictor = nn.Sequential(nn.Linear(out_features, out_features, bias=False),
                                            nn.BatchNorm1d(out_features),
                                            nn.ReLU(inplace=True), # hidden layer
                                            nn.Linear(out_features, out_features)) # output layer
        

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        # x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return x

    def forward_flatten(self, x: Tensor) -> Tensor:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        # x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.model.fc(x)

        return x

    def forward_ssl(self, x1: Tensor, x2: Tensor) -> Tensor:
        p1 = self.forward(x1)
        p2 = self.forward(x2)
        z1 = self.model.predictor(p1)
        z2 = self.model.predictor(p2)

        return p1, p2, z1, z2