# @Date    : 2020-12-10
# @Version : 0.2.1

# [github: pytorch_example, mnist](https://gitee.com/hainan89/pytorch-examples/blob/master/mnist/main.py)
# [pytorch: tutorial, mnist](https://pytorch.org/tutorials/beginner/nn_tutorial.html)

# %%
import ipyenv as uu
uu.enpy("torch")

# %%
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# %%
from collections import namedtuple
HyperParameters = namedtuple("HyperParameters", [
        "use_cuda",
        "save_model",
        "seed",  # random seed
        "batch_size",  # batch size for training
        "test_batch_size",  # batch size for testing
        "epochs",  # number of epochs to train
        "lr",  # learning rate
        "gamma",  # Learning rate step gamma
        "log_interval"])  # how many batches to wait before logging training status

args = HyperParameters(**{
    "use_cuda": True,
    "save_model": True,
    "seed": 1234,
    "batch_size": 64,
    "test_batch_size": 1000,
    "epochs": 4,
    "lr": 1.0,
    "gamma": 0.7,
    "log_interval": 10
})

# %% 载入数据集
torch.manual_seed(args.seed)

device = torch.device("cuda" if args.use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

# %% 让我们看看一批测试数据由什么组成
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_targets)
print(example_data.shape)

# %% 我们可以使用matplotlib来绘制其中的一些
import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

# %% 构建模型
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# %% 实例化模型
model = MnistNet().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

# %% 训练与测试函数
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()  # 反向传播
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# %% 训练
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()

# %% 保存模型
if args.save_model:
    torch.save(model.state_dict(), "./data/mnist_cnn.pt")
