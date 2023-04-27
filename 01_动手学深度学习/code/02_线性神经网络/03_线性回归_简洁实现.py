import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
# nn是神经⽹络的缩写
import torch.nn as nn

if __name__ == '__main__':
    if True:
        print("-------------------1. 生成数据集合----------------------")
        # 定义真实参数w和b
        # true_w为长度为2的张量，数值分别为2和-3.4
        true_w = torch.tensor([2, -3.4])
        # true_b为标量，数值为4.2
        true_b = 4.2

        # features和labels分别为特征和标签，使用Dive into Deep Learning中的synthetic_data函数生成1000个样本
        # 通过d2l.synthetic_data函数生成特征features和标签labels
        features, labels = d2l.synthetic_data(true_w, true_b, 1000)

        print("-------------------2. 读取数据集----------------------")
        # 定义一个加载数据的函数，其中data_arrays表示特征(features)和标签(labels)，batch_size表示每个小批量(batch)的样本数，is_train表示是否是训练集，默认为True
        def load_array(data_arrays, batch_size, is_train=True):  # @save
            """构造⼀个PyTorch数据迭代器"""
            # 使用TensorDataset函数将特征和标签合并成一个数据集
            dataset = data.TensorDataset(*data_arrays)
            # 使用DataLoader函数将数据集划分为一批一批的数据进行训练，返回一个DataLoader对象
            return data.DataLoader(dataset, batch_size, shuffle=is_train)

        # 设置每个小批量(batch)的样本数为10
        batch_size = 10
        # 调用load_array函数将数据进行划分
        data_iter = load_array((features, labels), batch_size)


        # 读取数据问题
        # iter_data = iter(data_iter)
        # print(next(iter_data))
        # print(next(iter_data))

        print("-------------------3. 定义模型----------------------")

        # 定义一个线性回归模型，其中第一个参数2表示特征的维度，第二个参数1表示输出的维度
        net = nn.Sequential(nn.Linear(2, 1))

        print("-------------------4. 初始化模型参数----------------------")
        # 使用正态分布随机初始化权重参数
        net[0].weight.data.normal_(0, 0.01)
        # 将偏置项参数初始化为0
        net[0].bias.data.fill_(0)

        print("-------------------5. 损失函数/优化算法----------------------")

        # 定义损失函数为均方误差损失函数（MSE）
        loss = nn.MSELoss()
        # 定义优化器为随机梯度下降优化器，其中net.parameters()表示只优化net中定义的参数
        trainer = torch.optim.SGD(net.parameters(), lr=0.03)

        print("-------------------6. 训练----------------------")

        # 训练模型，将数据划分为num_epochs个小批量(batch)进行训练
        num_epochs = 3
        for epoch in range(num_epochs):
            # 从数据迭代器中按批量(batch)取出数据进行训练
            for X, y in data_iter:
                # 求出模型的预测值net(X)和标签值y的均方误差
                l = loss(net(X), y)
                # 将梯度清零，避免梯度累加导致的结果错误
                trainer.zero_grad()
                # 反向传播求梯度
                l.backward()
                # 使用优化器更新权重和偏置项参数
                trainer.step()
            # 输出本轮训练的损失函数值
            l = loss(net(features), labels)
            print(f'epoch {epoch + 1}, loss {l:f}')

        # 获取训练后的权重参数和偏置项参数
        w = net[0].weight.data
        print('w的估计误差：', true_w - w.reshape(true_w.shape))
        b = net[0].bias.data
        print('b的估计误差：', true_b - b)
