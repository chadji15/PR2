import torch
import torch.nn as nn

class MyNet(nn.Module):
 
    def __init__(self, num_classes):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1, padding=0, bias=True)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=1,bias=True)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, stride=1, padding=1, bias=True)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=40, out_channels=100, kernel_size=5, stride=2, padding=1, bias=True)
        self.relu4 = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(in_features= 2500, out_features=num_classes)
        self.reluL = nn.ReLU()
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.reluL(x)
        x = self.logSoftmax(x)
        return x
