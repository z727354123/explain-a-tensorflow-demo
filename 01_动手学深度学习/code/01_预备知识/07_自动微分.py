import torch

if __name__ == '__main__':
    if True:
        print("-------------------反向传播----------------------")
        x = torch.arange(4.0);
        print('x = torch.arange(4.0)\n\tx={x}'.format(val=x.grad, x=x))

        print('x = torch.arange(4.0)\n\tx={x}'.format(val=x.grad, x=x))



        x.requires_grad_(True)
        print('x.requires_grad_(True)\n\tx={x}, x.grad={val}'.format(val=x.grad, x=x))


        y = 2 * torch.dot(x, x)
        print('y = 2 * torch.dot(x, x) \n\tx.grad={val} \n\ty={y}'.format(val=x.grad, y=y))

        y.backward();
        print('y.backward() \n\tx.grad={val} \n\ty={y}'.format(val=x.grad, y=y))

        val = x.grad == 4 * x
        print('x.grad == 4 * x: \n\t{val}'.format(val=val))



        print("-------------------reset _ not clear ----------------------")
        print('\tx.grad={val}'.format(val=x.grad, y=y))

        y = 2 * torch.dot(x, x)
        print('y = 2 * torch.dot(x, x) \n\tx.grad={val} \n\ty={y}'.format(val=x.grad, y=y))

        y.backward();
        print('y.backward() \n\tx.grad={val} \n\ty={y}'.format(val=x.grad, y=y))

        val = x.grad == 4 * x
        print('x.grad == 4 * x: \n\t{val}'.format(val=val))

        print("-------------------reset _ yes clear ----------------------")
        print('\tx.grad={val}'.format(val=x.grad, y=y))
        # x.grad.zero_();
        print('x.grad.zero_() \n\tx.grad={val}'.format(val=x.grad, y=y))

        y = 2 * torch.dot(x, x)
        print('y = 2 * torch.dot(x, x) \n\tx.grad={val} \n\ty={y}'.format(val=x.grad, y=y))

        y.backward();
        print('y.backward() \n\tx.grad={val} \n\ty={y}'.format(val=x.grad, y=y))

        val = x.grad == 4 * x
        print('x.grad == 4 * x: \n\t{val}'.format(val=val))

    if True:
        print("-------------------⾮标量变量的反向传播----------------------")
        x = torch.arange(4.0, requires_grad=True);
        print('torch.arange(4.0, requires_grad=True)\n\tx={x} \n\tx.grad={val}'.format(val=x.grad, x=x))

        y = x * x
        print('y = x * x \n\tx.grad={val} \n\ty={y}'.format(val=x.grad, y=y))
        y = y.sum();
        print('y = y.sum() \n\tx.grad={val} \n\ty={y}'.format(val=x.grad, y=y))
        y.backward();
        print('y.backward() \n\tx.grad={val} \n\ty={y}'.format(val=x.grad, y=y))



        print("-------------------⾮标量变量的反向传播----------------------")
        x = torch.arange(4.0, requires_grad=True);
        print('torch.arange(4.0, requires_grad=True)\n\tx={x} \n\tx.grad={val}'.format(val=x.grad, x=x))

        y = torch.dot(x, x)
        print('torch.dot(x, x) \n\tx.grad={val} \n\ty={y}'.format(val=x.grad, y=y))
        y.backward();
        print('y.backward() \n\tx.grad={val} \n\ty={y}'.format(val=x.grad, y=y))

    if True:
        print("-------------------分离计算----------------------")
        x = torch.arange(4.0, requires_grad=True);
        print('torch.arange(4.0, requires_grad=True)\n\tx={x} \n\tx.grad={val}'.format(val=x.grad, x=x))

        y = x * x
        print('y = x * x \n\tx.grad={val} \n\ty={y}'.format(val=x.grad, y=y))
        z = y * x
        print('z = y * x \n\tx.grad={val} \n\tz={z}'.format(val=x.grad, z=z))
        z.sum().backward();
        print('z.sum().backward() \n\tx.grad={val} \n\tz={z}'.format(val=x.grad, z=z))

        print('x.grad == y: {val}'.format(val=(x.grad == y)))
        print('x.grad == y.detach(): {val}'.format(val=(x.grad == y.detach())))
        print('x.grad == y * 3: {val}'.format(val=(x.grad == y * 3)))
        print('x.grad == y.detach() * 3: {val}'.format(val=(x.grad == y.detach() * 3)))
        print('x.grad == 2 * x: {val}'.format(val=(x.grad == 2 * x)))
        print('x.grad == 3 * x * x: {val}'.format(val=(x.grad == 3 * x * x)))
        print("-------------------分离计算----------------------")
        x = torch.arange(4.0, requires_grad=True);
        print('torch.arange(4.0, requires_grad=True)\n\tx={x} \n\tx.grad={val}'.format(val=x.grad, x=x))

        y = x * x
        u = y.detach()
        print('y = x * x \nu = y.detach() \n\tx.grad={val} \n\ty={y} \n\tu={u}'.format(val=x.grad, y=y, u=u))
        z = u * x
        print('z = u * x \n\tx.grad={val} \n\tz={z}'.format(val=x.grad, z=z))
        z.sum().backward();
        print('z.sum().backward() \n\tx.grad={val} \n\tz={z}'.format(val=x.grad, z=z))
        print('x.grad == y: {val}'.format(val=(x.grad == y)))
        print('x.grad == y.detach(): {val}'.format(val=(x.grad == y.detach())))
        print('x.grad == 2 * x: {val}'.format(val=(x.grad == 2 * x)))
        print('x.grad == 3 * x * x: {val}'.format(val=(x.grad == 3 * x * x)))







