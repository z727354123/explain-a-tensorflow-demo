# 导入PyTorch库
import torch
# 导入PyTorch视觉库
import torchvision
# 导入PyTorch数据工具库
from torch.utils import data
# 从PyTorch视觉库中导入数据转换工具
from torchvision import transforms
# 导入d2l库中的PyTorch模块
from d2l import torch as d2l

if __name__ == '__main__':
    if True:
        print("-------------------1. 生成数据集合----------------------")
        # 使用SVG格式显示图像
        d2l.use_svg_display()

        # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
        # 并除以255使得所有像素的数值均在0〜1之间
        # 定义一个将图像转换为Tensor格式的变换
        trans = transforms.ToTensor()

        # 加载FashionMNIST数据集，train=True表示加载训练集，train=False表示加载测试集
        # transform参数表示对数据集进行的变换，这里使用上面定义的trans将图像转换为Tensor格式
        # download=True表示如果数据集不存在则自动下载
        mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)

        # 打印训练集和测试集的大小
        print(len(mnist_train), len(mnist_test))

        # 打印第一张训练集图像的形状
        print(mnist_train[0][0].shape)


        # 定义一个函数用于获取FashionMNIST数据集
        def get_fashion_mnist_labels(labels):
            """返回Fashion-MNIST数据集的⽂本标签, 将数值标签转换为文本标签"""
            text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                           'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
            return [text_labels[int(i)] for i in labels]


        # 定义一个函数show_images，用于展示一组图片
        # 参数imgs表示要展示的图片集合
        # 参数num_rows表示要展示的行数
        # 参数num_cols表示要展示的列数
        # 参数titles表示每张图片的标题，可选
        # 参数scale表示图片的缩放比例，默认为1.5
        def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
            """绘制图像列表"""
            # 计算展示图片的总大小
            figsize = (num_cols * scale, num_rows * scale)
            # 创建一个子图
            _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
            # 将子图展平为一维数组
            axes = axes.flatten()
            # 遍历每张图片并展示
            for i, (ax, img) in enumerate(zip(axes, imgs)):
                # 如果图片是一个Tensor，则将其转换为numpy数组并展示
                if torch.is_tensor(img):
                    ax.imshow(img.numpy())
                else:
                    ax.imshow(img)
                # 隐藏x轴和y轴
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                # 如果有标题，则设置标题
                if titles:
                    ax.set_title(titles[i])
            # 返回展示的子图
            return axes


        # 加载MNIST数据集
        X, y = next(iter(data.DataLoader(mnist_train, batch_size=36)))
        # 展示36张图片，每行展示9张，共展示4行，标题为对应的标签
        X = X.reshape(36, 28, 28)
        X[0][2] = 0.9
        X[0][1] = 0.9
        # show_images(X, 4, 9, titles=get_fashion_mnist_labels(y));

        print("-------------------读取小批量----------------------")
        batch_size = 256


        def get_dataloader_workers():  # @save
            """使⽤4个进程来读取数据"""
            return 4


        print("-------------------startLoad----------------------")
        # 定义训练数据集的数据加载器train_iter，其中batch_size表示每个批次的数据量，shuffle=True表示每个epoch时打乱数据集，num_workers=get_dataloader_workers()表示使用get_dataloader_workers()函数返回的进程数来读取数据
        train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
        timer = d2l.Timer()
        # iter 是否多余 ?
        for X, y in iter(train_iter):
            continue
        print(f'{timer.stop():.2f} sec')

        print("-------------------整合组件----------------------")


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

        train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
        # iter 是否多余 ?
        for X, y in iter(train_iter):
            print(X.shape, X.dtype, y.shape, y.dtype)
            break
