import math

import torch

if __name__ == '__main__':

	if True:
		print("-------------------华丽分割线----------------------")
		x = torch.arange(3).reshape(3, 1)
		y = torch.ones(3).reshape(1, 3)
		print(x)
		print(y)
		print(y + x)

	if True:
		print("-------------------切片----------------------")
		x = torch.arange(12).reshape(3, 4)
		print(x)
		print(x[1, 2])
		print(x[1: 3, :])
		print(x[1: 3])

	if True:
		print("-------------------切片----------------------")
		x = torch.arange(12).reshape(3, 4)
		y = torch.arange(12).reshape(3, 4)
		d = torch.zeros_like(x)
		print('id(d):', id(d))
		d[:] = x + y
		print(d)
		print('id(d):', id(d))
