# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 程序选择框
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
ParX =('Par4.2', 'Par4.3', 'Par4.4')
ParX_val = ParX[0]
print(ParX_val+"被选中")


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 4.2 utils.data 简介
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
if ParX_val == 'Par4.2':
    # 1) 导入需要的模块
    import torch
    from torch.utils import data
    import numpy as np

    # 2) 定义获取数据集的类
    # 该类继承基类 Dataset，自定义一个数据集及对应标签
    class TestDataset(data.Dataset): # 继承 Dataset
        def __init__(self):
            self.Data = np.asarray([[1,2], [3,4], [2,1], [3,4], [4,5]]) # 一些由 2 维向量表示的数据集
            self.Label = np.asarray([0, 1, 0, 1, 2]) # 这是数据集对应的标签

        def __getitem__(self, index):
            # 把 numpy 转换为 Tensor
            txt = torch.from_numpy(self.Data[index])
            label = torch.tensor(self.Label[index])
            return txt,label
        
        def __len__(self):
            return len(self.Data)

    # 3) 获取数据集中的数据
    Test = TestDataset()
    # print(Test[2])
    # print(Test.__len__())

    if __name__ == "__main__":
        pass
        test_loader = data.DataLoader(Test,batch_size=2,shuffle=False,num_workers=2)
        # for i,traindata in enumerate(test_loader):
            # print('i:',i)
            # Data,Label=traindata
            # print('data:',Data)
            # print('Label:',Label)

        test_loader = data.DataLoader(Test,batch_size=2,shuffle=False,num_workers=2)
        dataiter = iter(test_loader)
        imgs, labels = next(dataiter)

        print(imgs, labels)


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 4.3 torchvision 简介
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
if ParX_val == 'Par4.3':
    # 4.3.2 ImageFolder
    from torchvision import transforms, utils
    from torchvision import datasets
    import torch
    import matplotlib.pyplot as plt

    my_trans = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    train_data = datasets.ImageFolder(r'D:\Users\Administrator\Desktop\PythonDLbasedonPytorch\data\torchvision_data', transform=my_trans)
    train_loader = data.DataLoader(train_data, batch_size=8, shuffle=True,)

    for i_batch, img in enumerate(train_loader):
        if i_batch == 0:
            print(img[1])
            fig = plt.figure()
            grid = utils.make_grid(img[0])
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.show()
            utils.save_image(grid, 'test01.png')
        break


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 4.4 可视化工具
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
if ParX_val == 'Par4.4':
    # 4.4.1 tensorboardX 简介
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir=r'D:\Users\Administrator\Desktop\PythonDLbasedonPytorch\Par4\logs')
    writer.add_xxx

    # 4.4.2 用 tensorboardX 可视化神经网络
    # 1) 导入需要的模块
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from torch.utils.tensorboard import SummaryWriter
    from tensorboardX import SummaryWriter
    import numpy as np
    # 2) 构建神经网络
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)
            self.bn = nn.BatchNorm2d(20)

        def forward(self, x):
            x = F.max_pool2d(self.conv1(x), 2)
            x = F.relu(x) + F.relu(-x)
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = self.bn(x)
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            x = F.softmax(x, dim=1)
            return x
    # 3) 把模型保存为 graph
    # 定义输入
    if __name__ == "__main__":
        pass
        input = torch.rand(32, 1, 28, 28)
        # 实例化神经网络
        model = Net()
        # 将 model 保存为 graph
        with SummaryWriter(log_dir=r'D:\Users\Administrator\Desktop\PythonDLbasedonPytorch\Par4\logs', comment='Net') as w:
            w.add_graph(model, (input, ))


    
    # 4.4.3 用 tensorboardX 可视化损失值
    dtype = torch.FloatTensor
    writer = SummaryWriter(log_dir='', comment='Linear')
    np.random.seed(100)
    x_train = np.linspace(-1, 1, 100).reshape(100, 1)
    y_train = 3 * np.power(x_train, 2) + 2 + 0.2 * np.random.rand(x_train.size).reshape(100, 1)

    model = nn.Linear(input_size, output_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoches):
        inputs = torch.from_numpy(x_train).type(dtype)
        targets = torch.from_numpy(y_train).type(dtype)

        output = model(inputs)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('TrainLoss', loss, epoch)


    

