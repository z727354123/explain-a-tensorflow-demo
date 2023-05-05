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
		train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

		print("-------------------1. 初始化模型参数----------------------")

		# PyTorch不会隐式地调整输⼊的形状。因此，
		# 我们在线性层前定义了展平层（flatten），来调整⽹络输⼊的形状
		net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


		def init_weights(m):
			if type(m) == nn.Linear:
				nn.init.normal_(m.weight, std=0.01)


		net.apply(init_weights);

		print("-------------------2. 重新审视Softmax的实现----------------------")
		loss = nn.CrossEntropyLoss(reduction='none')

		print("-------------------3. 优化算法----------------------")
		trainer = torch.optim.SGD(net.parameters(), lr=0.1)


		print("-------------------4. 训练----------------------")
		num_epochs = 10
		d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
		print("-------------------5. 预测----------------------")


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