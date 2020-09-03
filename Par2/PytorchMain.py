# -*- coding: utf-8 -*-

# 1) 导入需要的库
import torch as t
# %matplotlib inline
from matplotlib import pyplot as plt

# 2) 生成输入数据 x 及目标数据 y
# 设置随机数种子，生成同一份数据，以便用多种方法进行比较
t.manual_seed(100)
xx = t.linspace(-1, 1, 100, requires_grad=True).view(100, 1)
x = t.unsqueeze(t.linspace(-1, 1, 100), dim=1)
y = 3 * x.pow(2) + 2 + 0.2 * t.rand(x.size(),requires_grad=True).view(100, 1)

# 3) 查看 x,y 数据分布情况
# 画图
# plt.scatter(x.detach().numpy(), y.detach().numpy())
# plt.show()

# 4) 初始化权重参数
# 随机初始化参数
w1 = t.randn(1, 1, dtype = t.float, requires_grad=True)
b1 = t.zeros(1, 1, dtype = t.float, requires_grad=True)

# 5) 训练模型
lr = 0.001

for i in range(8):
    yy_pred = w1 * x + b1
    y_pred = x.pow(2).mm(w1) + b1

    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()
    loss.backward(retain_graph=True)
    # loss.backward()

    with t.no_grad():
        w1 -= lr * w1.grad
        b1 -= lr * b1.grad

        # w1 = w1 - lr * w1.grad
        # b1 = b1 - lr * b1.grad

        # grad_w = w1.grad
        # grad_b = b1.grad

        # w1 -= lr * grad_w
        # b1 -= lr * grad_b

        w1.grad.zero_()
        b1.grad.zero_()

    print(loss)
    print(loss,y)
