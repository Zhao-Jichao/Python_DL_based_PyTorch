# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 程序选择框
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
ParX =('Par8.1.2', 'Par8.1.3', 'Par8.3')
ParX_val = ParX[1]
print("正在运行第 "+ParX_val+" 节程序......")



# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Par8.1.2 变分自编码器
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
if ParX_val == 'Par8.1.2':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # 定义重构损失函数及 KL 散度
    # reconst_loss = F.binary_cross_entropy(x_renconst, x, size_average=False)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # 两者相加得总损失
    loss = reconst_loss + kl_div


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Par8.1.3 用变分自编码器生成图像
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
if ParX_val == 'Par8.1.3':
    # 1) 导入必要的包
    import os
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    from torchvision import transforms
    from torchvision.utils import save_image
    # 2) 定义一些超参数
    image_size = 784
    h_dim = 400
    z_dim = 20
    num_epochs = 30
    batch_size = 128
    learning_rate = 0.001
    # 3) 对数据进行预处理，转换为 Tensor，把数据集转换为循环、可批量加载的数据集
    # 下载 MNIST 数据集
    dataset = torchvision.datasets.MNIST(root='./data', 
                                        train=True, 
                                        transform=transforms.ToTensor(), 
                                        download=False)
    # 数据加载
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    # 4) 构建 AVE 模型，主要有 Encode 和 Decode 两部分组成
    # 定义 AVE 模型
    class VAE(nn.Module):
        def __init__(self, image_size=784, h_dim=400, z_dim=20):
            super(VAE, self).__init__()
            self.fc1 = nn.Linear(image_size, h_dim)
            self.fc2 = nn.Linear(h_dim, z_dim)
            self.fc3 = nn.Linear(h_dim, z_dim)
            self.fc4 = nn.Linear(z_dim, h_dim)
            self.fc5 = nn.Linear(h_dim, image_size)

        def encode(self, x):
            h = F.relu(self.fc1(x))
            return self.fc2(x), self.fc3(x)

        # 用 mu，log_var 生成一个潜在空间点 z。mu，log_var 为两个统计参数，我们假设这个假设分布能生成图像
        def reparameterize(self, mu, log_var):
            std = torch.exp(log_var/2)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            h = F.relu(self.fc4(z))
            return F.sigmoid(self.fc5(h))

        def forward(self, x):
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            x_renconst = self.decode(z)
            return x_renconst, mu, log_var
    # 5) 选择 GPU 及优化器
    # 设置 PyTorch 在哪块 GPU 上运行，这里假设使用序号为 1 的这块 GPU
    torch.cuda.set_device(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 6) 训练模型，同时保存原图像与随机生成的图像
    for epoch in range(10):
        with torch.no_grad():
            # 保存采样图像，及潜在向量 z 通过解码器生成的新图像
            z = torch.randn(batch_size, z_dim).to(device)
            out = model.decode(z).view(-1, 1, 28, 28)
            save_image(out, os.path.join('./testPar8', 'sampled-{}.png'.format(epoch+1)))
            # 保存重构图像，即原图像通过解码器生成的图像
            out, _, _ = model(x)
            x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
            save_image(x_concat, os.path.join('./testPar8', 'reconst-{}.png'.format(epoch+1)))
    # 7) 展示原图像及重构图像
    reconsPath = './ave_samples/reconst-30.png'
    Image = mpimg.imread(reconsPath)
    plt.imshow(Image)
    plt.axis('off')
    plt.show()
    # 8) 显示由潜在空间点 z 生成的新图像
    genPath = './ave_samples/sampled-30.png'
    Image = mpimg.imread(genPath)
    plt.imshow(Image)
    plt.axis('off')
    plt.show()


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Par8.3 用 GAN 生成图像
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
if ParX_val == 'Par8.3':
    # 构建判别器
    D = nn.Sequential(
        nn.Linear(image_size, hidden_size), 
        nn.LeakyReLU(0.2), 
        nn.Linear(hidden_size, hidden_size), 
        nn.LeakyReLU(0.2), 
        nn.Linear(hidden_size, 1), 
        nn.Sigmoid())
    # 生成器
    G = nn.Sequential(
        nn.Linear(latent_size, hidden_size), 
        nn.ReLU()
        nn.Linear(hidden_size, hidden_size), 
        nn.ReLU(), 
        nn.Linear(hidden_size, image_size), 
        nn.Tanh())
    # 训练模型
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            images = images.reshape(batch_size, -1).to(device)

            # 定义图像是真或假的标签
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, -1).to(device)
            
            # 训练判别器
            # 定义判别器对真图像的损失函数
            outputs = D(images)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            # 定义判别器对假图像（由潜在空间点生成的图像）的损失函数
