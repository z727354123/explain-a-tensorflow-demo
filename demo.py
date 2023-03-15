import tensorflow as tf  # 导入TensorFlow库
import matplotlib.pyplot as plt  # 导入matplotlib库并将其重命名为plt
mnist = tf.keras.datasets.mnist  # 加载MNIST数据集
data = mnist.load_data()  # 载入数据集并存储到变量data中

# 分别将训练数据和测试数据存储到x_train、y_train和x_test、y_test中
# 将训练集和测试集分别分配给x_train, y_train 和 x_test, y_test
(x_train, y_train),(x_test, y_test) =  mnist.load_data()


# 对训练数据和测试数据进行归一化处理
# 将训练集和测试集中的图像数据从0-255标准化到0-1之间
x_train, x_test = x_train / 255.0, x_test / 255.0

# 构建模型
model = tf.keras.models.Sequential([  # 创建神经网络模型
# 输入层，将28x28的图像矩阵展平成784个节点
    # 输入层，将28x28的图像展开成一维向量
  tf.keras.layers.Flatten(input_shape=(28, 28)),

    # 隐藏层，有128个神经元，采用ReLU激活函数
# 隐藏层1，使用ReLU激活函数，包含128个节点
  tf.keras.layers.Dense(128, activation='relu'),

    # Dropout层，防止过拟合
# 在模型的训练过程中，随机将一部分节点输出设置为0，以避免过拟合
  tf.keras.layers.Dropout(0.2),


    # 输出层，有10个神经元，采用Softmax激活函数
# 输出层，使用softmax激活函数，输出10个类别的概率分布
  tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型
              # 编译模型，选择优化器为Adam
# 优化器使用Adam算法
model.compile(optimizer='adam',

               # 损失函数使用稀疏分类交叉熵
               loss='sparse_categorical_crossentropy',

               # 衡量模型性能的指标是准确度
               metrics=['accuracy'])


model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)

# 可视化展示前25个训练图像

# 创建一个大小为10x10的图像
plt.figure(figsize=(10, 10))
for i in range(25):
    # 将图像划分为5行5列，i+1代表当前子图的位置
    plt.subplot(5, 5, i + 1)
    # 隐藏x轴标签
    plt.xticks([])
    # 隐藏y轴标签
    plt.yticks([])
    # 隐藏网格线
    # plt.grid(False)
    # 在子图中显示第i张图像
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    # 在子图下方显示图像的标签
    plt.xlabel(y_train[y_train[i]])
plt.show()  # 显示图像

