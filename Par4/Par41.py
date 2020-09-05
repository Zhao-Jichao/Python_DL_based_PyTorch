# ImageFolder 
import torch
from torch.utils import data
from torchvision import transforms, datasets, utils



train_trans = transforms.Compose([
    transforms.CenterCrop(1536/2), # 中心位置切割
    transforms.ToTensor(),
])

label_trans = transforms.Compose([
    transforms.CenterCrop(1536), # 中心位置切割
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder("/home/jichao/gitRes/Datasets/DIV2K/train", transform=train_trans)
train_loader = data.DataLoader(train_data, batch_size=8, shuffle=False)

label_data = datasets.ImageFolder("/home/jichao/gitRes/Datasets/DIV2K/label", transform=label_trans)
label_loader = data.DataLoader(label_data, batch_size=8, shuffle=False)

for i_batch, img in enumerate(train_loader):
    if i_batch == 0:
        utils.save_image(img[0], 'train.png')
    break

for i_batch, img in enumerate(label_loader):
    if i_batch == 0:
        utils.save_image(img[0], 'label.png')
    break

