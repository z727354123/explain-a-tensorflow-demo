import math

import torch

if __name__ == '__main__':
	ones = torch.ones(3)
	randn = torch.randn(3)
	print(ones)
	print(randn)
	print(ones + randn)

	if True:
		print("-------------------华丽分割线----------------------")
		x = torch.tensor([1.0, 2, 4, 8, 0])
		y = torch.tensor([math.e])
		print(torch.exp(x))
		print(y ** x)
		print(torch.sum(x))

	if True:
		print("-------------------张量连结----------------------")
		x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
		# y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
		y = torch.tensor([[2.0, 1, 4, 3]])
		print(x)
		print(y)
		print(torch.cat((x, y), dim=0))
		# print(torch.cat((x,y), dim=1))
		# print(torch.cat((x,y), dim=-1))
		print(torch.cat((x, y), dim=-2))
