import math
import time
import numpy as np
import torch
from d2l import torch as d2l


class Timer:  # @save
    """记录多次运⾏时间"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停⽌计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


if __name__ == '__main__':
    if True:
        print("-------------------方式1----------------------")
        n = 10000
        a = torch.ones([n])
        b = torch.ones([n])
        c = torch.zeros(n)
        timer = Timer()
        for i in range(n):
            c[i] = a[i] + b[i]
        print(f'{timer.stop():.10f} se')
        print(c)
    if True:
        print("-------------------方式2----------------------")
        n = 10000
        a = torch.ones([n])
        b = torch.ones([n])
        timer = Timer()
        d = a + b + b + a + b + b + a + b + b + a
        print(f'{timer.stop():.10f} se')
        print(d)
    if True:
        print("-------------------正态分布与平方损失----------------------")


        def normal(x, mu, sigma):
            p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
            return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)


        # 再次使⽤numpy进⾏可视化
        x = np.arange(-7, 7, 0.01)
        # 均值和标准差对
        params = [(0, 1), (0, 2), (3, 1)]
        d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
                 ylabel='p(x)', figsize=(4.5, 2.5),
                 legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
