# 导入PyTorch库
import torch
from torch import nn
from IPython import display
from d2l import torch as d2l
# 从PyTorch视觉库中导入数据转换工具
from torchvision import transforms
# 导入PyTorch视觉库
import torchvision

if __name__ == '__main__':
	if True:

		batch_size = 256
		# 加载Fashion-MNIST数据集，train_iter和test_iter分别是训练和测试数据集的迭代器
		train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

		# 1. 初始化模型参数
		print("-------------------1. 初始化模型参数----------------------")

		# PyTorch不会隐式地调整输⼊的形状。因此，
		# 我们在线性层前定义了展平层（flatten），来调整⽹络输⼊的形状
		# 初始化一个Sequential容器的神经网络，包含Flatten和一个全连接层
		net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


		# 初始化模型参数，此处为全连接层的权重
		# 初始化权重的函数
		def init_weights(m):
			if type(m) == nn.Linear:
				nn.init.normal_(m.weight, std=0.01)


		# 对模型的每个参数调用init_weights函数
		net.apply(init_weights);

		# 2. 重新审视Softmax的实现
		print("-------------------2. 重新审视Softmax的实现----------------------")
		# 定义损失函数CrossEntropyLoss，reduction='none'表示不对loss求平均
		loss = nn.CrossEntropyLoss(reduction='none')

		print("-------------------3. 优化算法----------------------")

		# 定义优化器SGD，学习率为0.1
		# 优化器，使用随机梯度下降法
		trainer = torch.optim.SGD(net.parameters(), lr=0.1)

		# 4. 训练
		print("-------------------4. 训练----------------------")

		# 训练的轮数
		num_epochs = 10

		# 训练模型
		d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

		# 5. 预测
		print("-------------------5. 预测----------------------")


		# 预测函数
		# 定义预测函数predict_ch3，用于预测测试集的图像类别，并展示前n张图像及其预测结果
		def predict_ch3(net, test_iter, n=6):
			for X, y in test_iter:
				break
			# 获取真实标签
			trues = d2l.get_fashion_mnist_labels(y)
			# 获取预测标签
			preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
			# 标题
			titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
			# 显示预测图片
			d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


		# 调用预测函数predict_ch3，展示前6张测试图像及其预测结果
		# 使用训练好的模型进行预测
		predict_ch3(net, test_iter)
