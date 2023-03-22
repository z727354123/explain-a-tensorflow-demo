import torch

if __name__ == '__main__':
	if True:
		print("-------------------华丽分割线----------------------")
		x = torch.arange(12)
		print(x)
		print(x.shape)
		print(x.numel())
	if True:
		print("-------------------华丽分割线----------------------")
		x = torch.arange(12)
		x = x.reshape(3, 4)
		print(x)
		print(x.shape)
		print(x.numel())
	if True:
		print("-------------------华丽分割线----------------------")
		x = torch.zeros(3, 4)
		print(x)
		print(x.shape)
		print(x.numel())
	if True:
		print("-------------------华丽分割线----------------------")
		x = torch.randn(3, 4)
		print(x)
		print(x.shape)
		print(x.numel())
	if True:
		print("-------------------华丽分割线----------------------")
		x = torch.tensor([[ 2.8967, -0.1779, -0.1320,  0.4614],
        [-0.0830,  1.4680,  1.6180,  1.1038],
        [-0.7441, -1.8332,  0.9546, -0.5217]])
		print(x)
		print(x.shape)
		print(x.numel())
