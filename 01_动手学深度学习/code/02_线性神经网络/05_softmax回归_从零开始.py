# 导入PyTorch库
import torch
from IPython import display
from d2l import torch as d2l
# 从PyTorch视觉库中导入数据转换工具
from torchvision import transforms
# 导入PyTorch视觉库
import torchvision

if __name__ == '__main__':
    if True:
        print("-------------------0. 生成数据集合----------------------")
        # 使用SVG格式显示图像
        d2l.use_svg_display()


        def get_dataloader_workers():  # @save
            """使⽤4个进程来读取数据"""
            return 4


        # 定义函数，加载Fashion-MNIST数据集到内存中
        # batch_size: 每个batch的大小
        # resize: 是否将图片大小调整为指定大小
        def load_data_fashion_mnist(batch_size, resize=None):  # @save
            '''下载Fashion-MNIST数据集，然后将其加载到内存中'''
            # 定义数据预处理操作
            trans = [transforms.ToTensor()]
            if resize:
                trans.insert(0, transforms.Resize(resize))
            trans = transforms.Compose(trans)
            # 加载训练集和测试集
            mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
            mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
            # 创建数据加载器
            train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                                                     num_workers=get_dataloader_workers())
            test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False,
                                                    num_workers=get_dataloader_workers())
            return train_iter, test_iter


        train_iter, test_iter = load_data_fashion_mnist(32, resize=28)

        print("-------------------1. 初始化模型参数----------------------")

        num_inputs = 784
        num_outputs = 10
        W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
        b = torch.zeros(num_outputs, requires_grad=True)

        print("-------------------2. 定义softmax操作----------------------")

        # 定义一个2x3的向量
        X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # 对二维矩阵按行或列求和，并在结果中保留行或列(keepdim=True)，这里分别求和的结果为1x3和2x1的向量
        print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))


        # 对于向量X的每个元素，计算其指数，然后归一化
        def softmax(X):
            # 对X中的每个元素做指数
            X_exp = torch.exp(X)
            # 对每个向量进行归一化，分母部分
            partition = X_exp.sum(1, keepdim=True)
            return X_exp / partition  # 这⾥应⽤了⼴播机制


        # 产生一个2x5的矩阵X，填充为标准正态分布
        X = torch.normal(0, 1, (2, 5))
        X_prob = softmax(X)
        print(X_prob, X_prob.sum(1))

        print("-------------------3. 定义模型----------------------")

        # 定义模型，这里W和b都是需要训练的参数，使用softmax作为激活函数
        def net(X):
            return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


        print("-------------------4. 定义损失函数----------------------")

        # 定义标签y
        y = torch.tensor([0, 2])
        # 定义模型预测的标签概率
        y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
        print(y_hat[[0, 1], y])

        # 使用下标[[0, 1], y]获取y_hat中标签所对应的概率
        # 交叉熵损失函数的定义，将y_hat中每个样本所预测分类概率中正确分类的概率取负数再求平均值
        def cross_entropy(y_hat, y):
            return - torch.log(y_hat[range(len(y_hat)), y])


        # 使用y_hat和y计算交叉熵
        print(cross_entropy(y_hat, y))

        print("-------------------5. 分类精度----------------------")


        def accuracy(y_hat, y):  # @save
            """计算预测正确的数量"""
            # 如果有多个预测值，则选择每个样本中预测概率最大的作为最终预测结果
            if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
                y_hat = y_hat.argmax(axis=1)
            # 将预测值与真实值做比较
            cmp = y_hat.type(y.dtype) == y
            # 统计预测对的数量
            return float(cmp.type(y.dtype).sum())

        # 计算y_hat和y的精度
        print(accuracy(y_hat, y) / len(y))


        def evaluate_accuracy(net, data_iter):  # @save
            """计算在指定数据集上模型的精度"""
            if isinstance(net, torch.nn.Module):
                net.eval()  # 将模型设置为评估模式
            metric = Accumulator(2)  # 正确预测数、预测总数
            with torch.no_grad():
                it = iter(data_iter)
                while True:
                    try:
                        X, y = next(it)  # 获取下一个元素
                        metric.add(accuracy(net(X), y), y.numel())
                    except StopIteration:  # 如果已经遍历完所有元素，则抛出 StopIteration 异常
                        break
                # for X, y in data_iter:
                #     metric.add(accuracy(net(X), y), y.numel())
            return metric[0] / metric[1]


        class Accumulator:  # @save
            """在n个变量上累加"""

            def __init__(self, n):
                self.data = [0.0] * n

            def add(self, *args):
                self.data = [a + float(b) for a, b in zip(self.data, args)]

            def reset(self):
                self.data = [0.0] * len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]


        print(evaluate_accuracy(net, test_iter))
        # ... 每次运行都会有不同结果
        print("-------------------6. 训练----------------------")


        def train_epoch_ch3(net, train_iter, loss, updater):  # @save
            """训练模型⼀个迭代周期（定义⻅第3章）"""
            # 将模型设置为训练模式
            if isinstance(net, torch.nn.Module):
                net.train()
            # 训练损失总和、训练准确度总和、样本数
            metric = Accumulator(3)
            for X, y in train_iter:
                # 计算梯度并更新参数
                y_hat = net(X)
                l = loss(y_hat, y)
                if isinstance(updater, torch.optim.Optimizer):
                    # 使⽤PyTorch内置的优化器和损失函数
                    updater.zero_grad()
                    l.mean().backward()
                    updater.step()
                else:
                    # 使⽤定制的优化器和损失函数
                    l.sum().backward()
                    updater(X.shape[0])
                metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
            # 返回训练损失和训练精度
            return metric[0] / metric[2], metric[1] / metric[2]


        class Animator:  # @save
            """在动画中绘制数据"""

            def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                         ylim=None, xscale='linear', yscale='linear',
                         fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                         figsize=(3.5, 2.5)):
                # 增量地绘制多条线
                if legend is None:
                    legend = []
                d2l.use_svg_display()
                self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
                if nrows * ncols == 1:
                    self.axes = [self.axes, ]
                # 使⽤lambda函数捕获参数
                self.config_axes = lambda: d2l.set_axes(
                    self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
                self.X, self.Y, self.fmts = None, None, fmts

            def add(self, x, y):
                # 向图表中添加多个数据点
                if not hasattr(y, "__len__"):
                    y = [y]
                n = len(y)
                if not hasattr(x, "__len__"):
                    x = [x] * n
                if not self.X:
                    self.X = [[] for _ in range(n)]
                if not self.Y:
                    self.Y = [[] for _ in range(n)]
                for i, (a, b) in enumerate(zip(x, y)):
                    if a is not None and b is not None:
                        self.X[i].append(a)
                        self.Y[i].append(b)
                self.axes[0].cla()
                for x, y, fmt in zip(self.X, self.Y, self.fmts):
                    self.axes[0].plot(x, y, fmt)
                self.config_axes()
                display.display(self.fig)
                display.clear_output(wait=True)


        def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
            """训练模型（定义⻅第3章）"""
            animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                                legend=['train loss', 'train acc', 'test acc'])
            for epoch in range(num_epochs):
                train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
            test_acc = evaluate_accuracy(net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))
            train_loss, train_acc = train_metrics
            assert train_loss < 0.5, train_loss
            assert train_acc <= 1 and train_acc > 0.7, train_acc
            assert test_acc <= 1 and test_acc > 0.7, test_acc


        lr = 0.1


        def updater(batch_size):
            return d2l.sgd([W, b], lr, batch_size)


        num_epochs = 10
        train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

        print("-------------------7. 预测----------------------")


        def predict_ch3(net, test_iter, n=6):  # @save
            """预测标签（定义⻅第3章）"""
            for X, y in test_iter:
                break
            trues = d2l.get_fashion_mnist_labels(y)
            preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
            titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
            d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

        # 用训练好的网络对前n个样本的输入输出进行预测
        predict_ch3(net, test_iter)
