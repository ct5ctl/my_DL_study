import torch
from torch import nn
from d2l import torch as d2l

if __name__ == '__main__':
    # 超参数
    batch_size = 64
    lr, num_epochs = 0.1, 10
    # 获取数据迭代器
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    # 定义模型及参数
    net = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 256), nn.ReLU(), nn.Linear(256, 10))
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01) # std是标准差
    net.apply(init_weights)
    # 选择损失函数
    loss = nn.CrossEntropyLoss(reduction="none")    # reduction="none"即要求求出的损失不要求均值
                                                    # （这里后面有求均值操作，故这里不要求，否则会使画的曲线数值过小而看不出来）
    # 选择优化函数
    trainer = torch.optim.SGD(net.parameters(), lr)
    # 训练模型
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    # 测试模型
    # d2l.predict_ch3(net, test_iter)
    d2l.plt.show()