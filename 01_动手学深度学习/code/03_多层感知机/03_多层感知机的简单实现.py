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
		train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
		net = nn.Sequential(nn.Flatten(),
		                    nn.Linear(784, 256),
		                    nn.ReLU(),
		                    nn.Linear(256, 10))


		def init_weights(m):
			if type(m) == nn.Linear:
				nn.init.normal_(m.weight, std=0.01)

		net.apply(init_weights);
		batch_size, lr, num_epochs = 256, 0.1, 10
		loss = nn.CrossEntropyLoss(reduction='none')
		trainer = torch.optim.SGD(net.parameters(), lr=lr)
		train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
		d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
