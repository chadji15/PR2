'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *
from utils import progress_bar
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

test_metrics = []
train_metrics = []
num_epoch = 60
step_lr = 0.04
low_lr = round(0.1 - step_lr,2)
high_lr = round(0.1 + step_lr,2)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # if(batch_idx in [0,60,128]):
        #     print(outputs.shape)
        #     plt.imshow(outputs.cpu().detach().numpy())
        #     plt.savefig('/content/gdrive/MyDrive/PR2/featuremap-{}.png'.format(batch_idx))
        #     plt.show()

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return train_loss


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        #torch.save(state, './checkpoint/ckpt.pth')
        torch.save(state, './checkpoint/ckpt.pth')

        best_acc = acc
    return acc

if __name__ == "__main__":
    fig1 = plt.figure(1)
    ax1 = fig1.gca()
    
    fig2 = plt.figure(2)
    ax2 = fig2.gca()
    learning_rate = high_lr
    print('\n\n Learning Rate: {}'.format(learning_rate))
    
    # Model
    print('==> Building model..')
    net = MyNet(10)
    net = net.to(device)
    
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                        momentum=0.9, weight_decay=5e-4)
                        
    start_plot = start_epoch
    train_metrics = []
    test_metrics = []
    for epoch in range(start_epoch, start_epoch+num_epoch):
        train_metrics.append(train(epoch))
        test_metrics.append(test(epoch))
        #scheduler.step()
        
    ax1.plot(range(start_plot, start_plot+num_epoch), train_metrics, label="LR {}".format(round(learning_rate, 2)))
    ax2.plot(range(start_plot, start_plot+num_epoch), test_metrics, label="LR {}".format(round(learning_rate, 2)))

    
    ax1.set_ylabel('Train metrics (Loss)')
    ax1.set_xlabel('Epochs')
    ax1.set_title('Train Metrics Different Learning Rates')
    fig1 = ax1.figure
    fig1.legend(loc='upper center')
    fig1.savefig('/content/gdrive/MyDrive/PR2/train-metrics-MyNet.png')
    print('Plot saved in train-metrics-MyNet.png')
    
    ax2.set_ylabel('Test metrics (Accuracy)')
    ax2.set_xlabel('Epochs')
    ax2.set_title('Test Metrics Different Learning Rates')
    fig2 = ax2.figure
    fig2.legend(loc='upper center')
    fig2.savefig('/content/gdrive/MyDrive/PR2/test-metrics-MyNet.png')
    print('Plot saved in test-metrics-MyNet.png')