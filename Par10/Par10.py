# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 程序选择框
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
ParX =('Par10.2', 'Par10.5')
ParX_val = ParX[0]
print("正在运行第 "+ParX_val+" 节程序......")


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Par10.2 特征提取
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
if ParX_val == 'Par10.2':

    # 1. 导入模块
    import torch
    from torch import nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    from torchvision import models
    from torchvision.datasets import ImageFolder
    from datetime import datetime

    # 2. 加载数据
    