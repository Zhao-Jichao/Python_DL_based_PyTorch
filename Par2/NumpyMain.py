# -*- coding: utf-8 -*-

# 1) 导入需要的库
import numpy as np
# %matplotlib inline
from matplotlib import pyplot as plt

# 2) 生成输入数据 x 及目标数据 y
# 设置随机数种子，生成同一份数据，以便用多种方法进行比较
np.random.seed(100)
x = np.linspace(-1, 1, 100).reshape(100, 1)
y = 3*np.power(x, 2) +2 + 0.2*np.random.rand(x.size).reshape(100,1)

# 3) 查看 x,y 数据分布情况
# 画图
# plt.scatter(x, y)
# plt.show()

# 4) 初始化权重参数
# 随机初始化参数
w1 = np.random.rand(1, 1)
b1 = np.random.rand(1, 1)

# 5) 训练模型
lr = 0.001

for i in range(800):
    y_pred = np.power(x,2) * w1 + b1

    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()

    grad_w = np.sum((y_pred - y) * np.power(x, 2))
    grad_b = np.sum((y_pred - y))

    w1 -= grad_w * lr
    b1 -= grad_b * lr

    # print(loss)
print(w1,b1)
plt.plot(x, y_pred, 'r-', label='predict')
plt.scatter(x, y, color='blue', marker='o', label='true')
plt.xlim(-1, 1)
plt.ylim(2, 6)
plt.legend()
plt.show()



