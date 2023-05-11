# 导入相关模块
import torch
from torch import nn
from d2l import torch as d2l

if __name__ == '__main__':
	# 加载数据集
	if True:
		# 设置批量大小
		batch_size = 256
		# 加载 Fashion-MNIST 数据集
		train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 从 d2l 中导入 Fashion-MNIST 数据集加载模块，加载训练集和测试集
		# 设置神经网络结构
		net = nn.Sequential(nn.Flatten(),  # 定义将图像展平，将输入大小为 (batch_size, 1, 28, 28) 的小批量样本 x 的形状转换为 (batch_size, 784) 的形状
		                    nn.Linear(784, 256),  # 输入大小为 784，输出大小为 256 的线性层
		                    nn.ReLU(),  # ReLU 激活函数
		                    nn.Linear(256, 10))  # 输入大小为 256，输出大小为 10 的线性层


		# 自定义初始化函数，对隐藏层使用正态分布随机初始化
		def init_weights(m):
			if type(m) == nn.Linear:
				nn.init.normal_(m.weight, std=0.01)


		# 对模型net的权重进行自定义初始化
		net.apply(init_weights);
		batch_size, lr, num_epochs = 256, 0.1, 10
		# 损失函数为交叉熵损失
		loss = nn.CrossEntropyLoss(reduction='none')
		trainer = torch.optim.SGD(net.parameters(), lr=lr)
		# 加载 Fashion-MNIST 数据集
		train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
		# 训练并评估模型net
		d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
		# 对测试集进行测试，并打印准确率
		d2l.predict_ch3(net, test_iter)

