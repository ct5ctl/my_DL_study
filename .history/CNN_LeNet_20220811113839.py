import imp
import torch
from torch import nn
from d2l import torch as d2l
import myTrainer

# 定义网络
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.Sigmoid(),       # 那是没有ReLU，用ReLU会更好
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

# 查看卷积层到全连接层时，第一个全连接层的输入特征数
# X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape: \t', X.shape)


# 超参数设置
lr, num_epochs, batch_size = 1.5, 10, 256
# 获取数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 训练
myTrainer.train_CNN(net, train_iter, test_iter, num_epochs, lr, device=d2l.try_gpu())
d2l.plt.show()
# 保存模型参数
torch.save(net.state_dict(), 'LeNet.params')