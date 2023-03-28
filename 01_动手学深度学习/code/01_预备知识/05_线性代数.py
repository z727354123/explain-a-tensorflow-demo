import os
import pandas as pd
import torch


if __name__ == '__main__':
    if True:
        print("-------------------矩阵转换----------------------")
        x = torch.arange(6).reshape(1, 2, 3)
        print(x)
        print(x.T)
        x = torch.arange(6).reshape( 2, 3)
        print(x)
        print(x.T)

    if True:
        print("-------------------张量----------------------")
        x = torch.arange(6).reshape(1, 2, 3)
        print(x)
        y = x.clone()
        print(x * y)
        x = torch.arange(6).reshape(1, 2, 3)
        y = x.clone().T
        print(x * y)



