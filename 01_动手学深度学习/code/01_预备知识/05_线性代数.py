import os
import pandas as pd
import torch

if __name__ == '__main__':
    if True:
        print("-------------------矩阵转换----------------------")
        x = torch.arange(6).reshape(1, 2, 3)
        print(x)
        # print(x.T)
        x = torch.arange(6).reshape(2, 3)
        print(x)
        print(x.T)

    if True:
        print("-------------------张量----------------------")
        # x = torch.arange(6).reshape(1, 2, 3)
        # print(x)
        # y = x.clone()
        # print(x * y)
        # x = torch.arange(6).reshape(1, 2, 3)
        # y = x.clone().T
        # print(x * y)

    if True:
        print("-------------------降维----------------------")
        x = torch.arange(6).reshape(1, 2, 3)
        print(x.sum())
        print(x.sum())

    if True:
        print("-------------------范数----------------------")
        val = torch.ones((4, 9))
        print(val)
        print(torch.norm(val))
        print(torch.norm(val, 1))
        print(torch.norm(val, 2))
        print(torch.norm(val, 3))

    if True:
        print("-------------------除法----------------------")
        val = torch.ones((3, 3))
        val2 = torch.arange(9).reshape(3, 3)
        val2[0, 0] = 1
        valSum = val2.sum(axis=1)
        print(val)
        print(val2)
        print(valSum)
        print(val2.sum(axis=0))
        print(val / val2)
        print(val / valSum)
    if True:
        print("-------------------2, 3, 4----------------------")
        val = torch.arange(24).reshape(2, 3, 4)
        print(val)
        print(val.sum(axis=0))
        print(val.sum(axis=1))
        print(val.sum(axis=2))
