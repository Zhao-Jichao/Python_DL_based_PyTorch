# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 程序选择框
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
ParX =('Par6.1', 'Par6.5', 'Par6.6', 'Par6.7')
ParX_val = ParX[2]
print("正在运行第 "+ParX_val+" 节程序......")


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Par6.1 卷积神经网络简介
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
if ParX_val == 'Par6.1':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class CNNNet(nn.Module):
        def __init__(self):
            super(CNNNet, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(1296, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool2(x)
            # print(x.shape)
            x = x.view(-1, 36*6*6)
            x = self.fc1(1296, 128)
            x = F.relu(x)
            x = self.fc2(128, 10)
            x = F.relu(x)
            return x

    net = CNNNet()
    net = net.to(device=device)


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Par6.5 PyTorch 实现 CIFAR-10 多分类
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
if ParX_val == 'Par6.5':
    # 1) 导入库及下载数据
    import torch
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)


    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)


    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truce')

    # 2) 随机查看部分数据
    import matplotlib.pyplot as plt
    import numpy as np

    # 显示图像
    def imshow(img):
        img = img / 2 + 0.5 #Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # 随机获取部分训练数据
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # 显示图像
    imshow(torchvision.utils.make_grid(images))
    # 打印标签
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # 6.5.3 构建网络
    # 1) 构建网络
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")

    class CNNNet(nn.Module):
        def __init__(self):
            super(CNNNet, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(1296, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool2(x)
            x = x.view(-1, 36*6*6)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            return x
    net = CNNNet()
    net = net.to(device=device)
    # 2) 查看网络结构
    # 显示网络中定义了哪些层
    print(net)
    # 3) 查看网络中前几层
    print(nn.Sequential(*list(net.children())[:4]))
    # 4) 初始化参数
    # for m in net.modules():
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.normal_(m.weight)
    #         nn.init.xavier_normal_(m.weight)
    #         nn.init.kaiming_normal_(m.weight) # 卷积层参数初始化
    #         nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Linear):
    #         nn.init.normal_(m.weight) # 全连接层参数初识化

    # 6.5.4 训练模型
    # 1) 选择优化器
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # 2) 训练模型
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取训练数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # 权重参数梯度清零
            optimizer.zero_grad()
            # 正向及反向传播
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 显示损失值
            running_loss += loss.item()
            if i % 500 == 499:    # Print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch+1, i+1, running_loss/500))
                running_loss = 0.0
        print("Finished Training")

    # 6.5.5 测试模型
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))
    # 各种类别的准确率
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %2d %%' %(classes[i], 100*class_correct[i]/class_total[i]))

    # 6.5.6 采用全局平均池化
    import torch.nn as nn
    import torch.nn.functional as F
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 5)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 36, 5)
            # self.fc1 = nn.Linear(16*16*5, 120)
            self.pool2 = nn.MaxPool2d(2, 2)
            # 使用全局平均池化层
            self.aap = nn.AdaptiveAvgPool2d(1)
            self.fc3 = nn.Linear(36, 10)

        def forward(self, x):
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.aap(x)
            x = x.view(x.shape[0], -1)
            x = self.fc3(x)
            return x
    net = Net()
    net = net.to(device)
    # 查看参数总量
    print("net_gvp have {} paramerters in total".format(sum(x.numel() for x in net.parameters())))


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Par6.6 模型集成提升性能
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
if ParX_val == 'Par6.6':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    # 6.6.1 使用模型
    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Conv2d(6, 16, 5)
            self.pool4 = nn.MaxPool2d(2, 2)
            self.fc5 = nn.Linear(16*5*5, 120)
            self.fc6 = nn.Linear(120, 84)
            self.fc7 = nn.Linear(84, 10)
        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = F.relu(x)
            x = self.pool4(x)
            x = x.view(16*5*5. -1)
            x = F.relu(self.fc5(x))
            x = F.relu(self.fc6(x))
            x = self.fc7(x)
            return x
    # 6.6.2 集成方法
    mlps = [net1.to(device), net2.to(device), net3.to(device)]
    optimizer = torch.optim.Adam([{"params":mlp.parameters()} for mlp in mlps], lr=LR)
    loss_function = nn.CrossEntropyLoss()

    for ep in range(EPOCHES):
        for img, label in trainloader:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad() # 10 个网络清除梯度
            for mlp in mlps:
                mlp.train()
                out = mlp(img)
                loss = loss_function(out, label)
                loss.backward() # 网络获得的梯度
            optimizer.step()

        pre = []
        vote_correct = 0
        mlps_correct = [0 for i in range(len(mlps))]
        for img,label in testloader:
            img,label = img.to(device), label.to(device)
            for i, mlp in enumerate(mlps):
                mlp.eval()
                out = mlp(img)

                _,prediction = torch.max(out, 1) # 按行取最大值
                pre_num = prediction.cpu().numpy()
                mlps_correct[i] += (pre_num==label.cpu().numpy()).sum()

                pre.append(pre_num)
            arr = np.array(pre)
            pre.clear()
            result = [Counter(arr[:,i]).most_common(1)[0][0] for  i in range(BATCHISIZE)]
            vote_correct += (result==label.cpu().numpy()).sum()
        print("epoch:" + str(ep) + "集成模型的正确率" + str(vote_correct/len(testloader)))

        for idx,coreect in enumerate(mlps_correct):
            print("模型" + str(idx) + "的正确率为:" + str(coreect/len(testloader)))
            

# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Par6.7 使用现代经典模型提升性能
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
if ParX_val == 'Par6.7':
    cfg = {
        'vgg16':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'vgg19':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }
    class VGG(nn.Module):
        def __init__(self, vgg_name):
            super(VGG, self).__init__()
            self.features = self._make_layers(cfg[vgg_name])
            self.classifier = nn.Linear(512, 10)

        def forward(self, x):
            out = self.features(x)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out
        
        def _make_layers(self, cfg):
            layers = []
            in_channels = 3
            for x in cfg:
                if x=='M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x),
                               nn.ReLU(inplace=True)]
                    in_channels = x
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
            return nn.Sequential(*layers)
    # VGG16 = VGG('VGG16')
