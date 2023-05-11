# 导入相关模块
import torch
from torch import nn
from d2l import torch as d2l

if __name__ == '__main__':
	# 加载数据集
	if True:
		print("-------------------初始化模型参数----------------------")
		# 设置批量大小
		batch_size = 256
		# 加载 Fashion-MNIST 数据集
		train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
		# 初始化模型参数
		print("-------------------1.初始化模型参数----------------------")
		# 输入特征数为784，输出特征数为10，隐含层输出个数为256
		num_inputs, num_outputs, num_hiddens = 784, 10, 256
		# 随机初始化第一个全连接层的权重
		W1 = nn.Parameter(torch.randn(
			num_inputs, num_hiddens, requires_grad=True) * 0.01)
		# 初始化第一个全连接层的偏差向量
		b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
		# 随机初始化第二个全连接层的权重
		W2 = nn.Parameter(torch.randn(
			num_hiddens, num_outputs, requires_grad=True) * 0.01)
		# 初始化第二个全连接层的偏差向量
		b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
		# 将所有参数放在一起便于更新
		params = [W1, b1, W2, b2]

		# 定义激活函数
		print("-------------------2.激活函数----------------------")


		def relu(X):
			# 初始化与输入形状相同的全0张量a
			a = torch.zeros_like(X)
			# a中对应位置的元素若小于输入中对应位置的元素,则 a 中对应位置取 0，否则取输入中对应位置的元素值。
			return torch.max(X, a)


		# 定义模型
		print("-------------------3.模型----------------------")


		def net(X):
			# 将输入转化为2D数组
			X = X.reshape((-1, num_inputs))
			# 进行第一层全连接
			H = relu(X @ W1 + b1)  # 这⾥“@”代表矩阵乘法
			# 输出层
			return (H @ W2 + b2)


		# 定义损失函数
		print("-------------------4.损失函数----------------------")
		# 交叉熵损失函数，且求得的结果不进行标准化（即不除以样本数）
		loss = nn.CrossEntropyLoss(reduction='none')

		# 训练模型
		print("-------------------5.训练----------------------")
		# 设置迭代周期为10，学习率为0.1
		num_epochs, lr = 10, 0.1
		# 它用小批量随机梯度下降训练模型参数
		updater = torch.optim.SGD(params, lr=lr)
		d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

		# 测试模型
		print("-------------------6.测试----------------------")
		# 对测试集进行测试，并打印准确率
		d2l.predict_ch3(net, test_iter)
