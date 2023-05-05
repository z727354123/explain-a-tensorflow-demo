import math
import time
import random
import numpy as np
import torch
from d2l import torch as d2l

if __name__ == '__main__':
    if True:
        print("-------------------1. 生成数据集合----------------------")


        # @save ：参数为w，b，num_examples
        def synthetic_data(w, b, num_examples):
            '''⽣成y=Xw+b+噪声'''
            # 生成符合正态分布的数据，均值为0，方差为1，生成（num_examples, len(w)）的矩阵X
            X = torch.normal(0, 1, (num_examples, len(w)))
            # 执行矩阵乘法，得到（num_examples, 1）的矩阵y，并加上偏置量b
            y = torch.matmul(X, w) + b
            cloneY = y.clone()
            # 生成符合正态分布的数据，均值为0，方差为0.01，生成与y矩阵大小一致的噪声元素，并将其累加到y矩阵中
            y += torch.normal(0, 0.01, y.shape)
            # 返回数据集矩阵X和标签矩阵y，其中标签矩阵y要修改其形状为（num_examples，1）的矩阵
            return X, y.reshape((-1, 1)), cloneY.reshape((-1, 1))


        # 定义一个2维张量，并给定值，作为线性函数真实w参数值
        true_w = torch.tensor([2, -3.4])
        true_b = 4.2  # 真实偏置
        # 调用生成函数synthetic_data，并给定参数值true_w, true_b, 1000。保存为数据集features和标签labels
        features, labels, realLabels = synthetic_data(true_w, true_b, 1000)
        # 打印第一条数据及其标签
        print('features:', features[0], '\nlabel:', labels[0])
        print('\tfeatures:', features[1], '\n\tlabel:', labels[1])

        # 画图
        # d2l.set_figsize()
        # d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
        print("-------------------2. 读取数据集----------------------")


        # 这是一个数据迭代器的函数，用于读取训练数据
        def data_iter(batch_size, features, labels):
            # 获取样本数量
            num_examples = len(features)
            # 创建样本下标的列表
            indices = list(range(num_examples))
            # 这些样本是随机读取的，没有特定的顺序
            random.shuffle(indices)
            # 循环迭代每个 batch 的样本
            for i in range(0, num_examples, batch_size):
                # 获得当前 batch 中的样本下标
                batch_indices = torch.tensor(
                    indices[i: min(i + batch_size, num_examples)])
                # 生成当前 batch 的特征和标签
                yield features[batch_indices], labels[batch_indices]


        batch_size = 4
        for X, y in data_iter(batch_size, features, labels):
            print(X, '\nt', y)
            break

        print("-------------------3. 初始化模型参数----------------------")
        w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)

        print("-------------------4. 定义模型/损失函数/优化算法----------------------")


        # 定义模型
        def linreg(X, w, b):
            """线性回归模型"""
            return torch.matmul(X, w) + b


        # 定义损失函数
        def squared_loss(y_hat, y):
            """均⽅损失"""
            return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


        # 定义优化算法
        def sgd(params, lr, batch_size):
            '''
            小批量随机梯度下降
            :param params: 待训练参数
            :param lr: 学习率
            :param batch_size: 批次大小
            '''
            with torch.no_grad():
                for param in params:
                    print(f"param={param} param.grad = {param.grad}")
                    param -= lr * param.grad / batch_size
                    # 更新参数的值
                    param.grad.zero_()
                    # 清空梯度，避免重复计算梯度影响优化效果


        print("-------------------5. 训练----------------------")

        lr = 0.03  # 学习率
        num_epochs = 3  # epoch数，即整个数据集迭代次数
        net = linreg  # 线性回归网络模型，可以自定义
        loss = squared_loss  # 损失函数，可以自定义
        for epoch in range(num_epochs):
            for X, y in data_iter(batch_size, features, labels):
                # X和y的小批量损失。计算当前批次数据的损失
                l = loss(net(X, w, b), y)
                # 因为l形状是(batch_size,1)，而不是一个标量。
                # l中的所有元素被加到一起，并以此计算关于[w,b]的梯度
                # 对当前取出的小批量数据的损失进行反向传播，计算[w, b]的梯度
                print(f"l.sum()={l.sum()}")
                l.sum().backward()
                # 使用参数的梯度更新参数。调用定义好的sgd优化算法进行参数更新
                sgd([w, b], lr, batch_size)
            with torch.no_grad():
                train_l = loss(net(features, w, b), labels)  # 在当前迭代次数结束后，计算整个训练集上的损失
                print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')  # 打印当前epoch数和整个训练集上的平均损失
                train_l = loss(net(features, w, b), realLabels)  # 在当前迭代次数结束后，计算整个训练集上的损失
                print(f'epoch2 {epoch + 1}, loss {float(train_l.mean()):f}')  # 打印当前epoch数和整个训练集上的平均损失

        print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')  # 打印w的估计误差
        print(f'b的估计误差: {true_b - b}')  # 打印b的估计误差
