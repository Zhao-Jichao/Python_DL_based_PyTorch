import os
import datetime

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image


if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MNIST('./data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

dataIndex = 0
starttime = datetime.datetime.now()

for epoch in range(num_epochs):
    for data in dataloader:
        dataIndex += 1
        # print('dataIndex:' + str(dataIndex))
        img, label = data
        # save_image(img, './mlp_img/imageData.png')
        img = img.view(img.size(0), -1)
        # save_image(img, './mlp_img/imageView.png')
        img = Variable(img).cuda()
        # save_image(img, './mlp_img/imageVar.png')
        # ===================forward=====================
        output = model(img)
        # save_image(img, './mlp_img/imageOutput.png')
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    endtime = datetime.datetime.now()
    print('epoch [{}/{}], loss:{:.4f}, time:{:.2f}s'.format(epoch + 1, num_epochs, loss.item(), (endtime-starttime).seconds))

    # if epoch % 10 == 0:
    pic = to_img(output.cpu().data)
    save_image(pic, './mlp_img/image_{}.png'.format(epoch))
    # save_image(output, './mlp_img/image_{}.png'.format(epoch))

torch.save(model.state_dict(), './sim_autoencoder.pth')