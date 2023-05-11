import torch
from d2l import torch as d2l

if __name__ == '__main__':
    if True:
        print("-------------------ReLU函数----------------------")
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        y = torch.relu(x)
        d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
    if True:
        print("-------------------ReLU函数的导数----------------------")
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        y = torch.relu(x)
        y.backward(torch.ones_like(x), retain_graph=True)
        d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))

    if True:
        print("-------------------sigmoid函数----------------------")
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        y = torch.sigmoid(x)
        d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))

    if True:
        print("-------------------sigmoid函数的导数----------------------")
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        y = torch.sigmoid(x)
        y.backward(torch.ones_like(x), retain_graph=True)
        d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid(x)', figsize=(5, 2.5))

    if True:
        print("-------------------tanh函数----------------------")
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        y = torch.tanh(x)
        d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))

    if True:
        print("-------------------tanh函数的导数----------------------")
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        y = torch.tanh(x)
        y.backward(torch.ones_like(x), retain_graph=True)
        d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh(x)', figsize=(5, 2.5))

