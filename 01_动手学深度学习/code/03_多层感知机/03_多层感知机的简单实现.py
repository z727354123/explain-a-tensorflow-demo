# 导入相关模块
import torch  # 导入 torch 模块
from torch import nn  # 从 torch 中导入 nn 模块
from d2l import torch as d2l  # 导入 d2l 中的 torch 模块

if __name__ == '__main__':
	# 加载数据集
	if True:
		# 设置批量大小
		batch_size = 256  # 设置批量大小为 256
		# 加载 Fashion-MNIST 数据集
		train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 从 d2l 中导入 Fashion-MNIST 数据集加载模块，加载训练集和测试集
		# 设置神经网络结构
		net = nn.Sequential(nn.Flatten(),  # 定义将图像展平，将输入大小为 (batch_size, 1, 28, 28) 的小批量样本 x 的形状转换为 (batch_size, 784) 的形状
		                    nn.Linear(784, 256),  # 输入大小为 784，输出大小为 256 的线性层
		                    nn.ReLU(),  # ReLU 激活函数
		                    nn.Linear(256, 10))  # 输入大小为 256，输出大小为 10 的线性层


		def init_weights(m):  # 自定义参数初始化方法，对于模型中的 nn.Linear 模块，将其权重参数初始化为在一个小的正态分布随机样本。
			if type(m) == nn.Linear:
				nn.init.normal_(m.weight, std=0.01)

		net.apply(init_weights);  # 对模型应用自定义参数初始化方法
		batch_size, lr, num_epochs = 256, 0.1, 10
		loss = nn.CrossEntropyLoss(reduction='none')
		trainer = torch.optim.SGD(net.parameters(), lr=lr)  # 定义优化器
		train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 重新加载 Fashion-MNIST 数据集
		d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)  # 调用模型训练函数进行训练