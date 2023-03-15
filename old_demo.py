# 导入所需的模块和库
import tensorflow as tf
import matplotlib.pyplot as plt

# 加载手写数字MNIST数据集
mnist = tf.keras.datasets.mnist

# 加载数据集并将其分为训练集和测试集
data = mnist.load_data()
(x_train, y_train), (x_test, y_test) =  mnist.load_data()

# 将数据集中的所有像素值除以255，以使它们在0到1之间
x_train, x_test = x_train / 255.0, x_test / 255.0

# 建立模型，使用Sequential模型并添加各个图层
model = tf.keras.models.Sequential([
  # 将输入数据展平成一维张量
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  # 添加第一个隐藏层，使用128个神经元，ReLU激活函数
  tf.keras.layers.Dense(128, activation='relu'),
  # 添加Dropout正则化层，以减少过拟合
  tf.keras.layers.Dropout(0.2),
  # 添加输出层，使用10个神经元，softmax激活函数，用于对输入数据进行分类
  tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型，定义损失函数、优化器和评估指标
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型，使用训练集数据进行拟合
model.fit(x_train, y_train, epochs=10)

# 评估模型，使用测试集数据进行评估，并返回损失和准确度
model.evaluate(x_test, y_test)
