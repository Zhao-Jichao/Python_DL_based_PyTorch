# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# 程序选择框
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
ParX =('Par9.3', 'Par.9.5')
ParX_val = ParX[1]
print("正在运行第 "+ParX_val+" 节程序......")


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Par9.3 特征提取
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
if ParX_val == 'Par9.3':
    import torch
    import torch.nn as nn
    import  torch.nn.functional as F
    
    # ArcFace
    class ArcMarginProduct(nn.Module):
        r"""Implement of large margin arc distance: :
            Args:
                in_features: size of each input sample
                out_features: size of each output sample
                s: norm of input feature
                m: margin

                cos(theta + m)
            """
        def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
            super(ArcMarginProduct, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.s = s
            self.m = m
            # 初始化权重
            self.weight = Parameter(torch.FloatTensor(out_features, in_features))
            nn.init.xavier_uniform_(self.weight)

            self.easy_margin = easy_margin
            self.cos_m = math.cos(m)
            self.sin_m = math.sin(m)
            self.th = math.cos(math.pi - m)
            self.mm = math.sin(math.pi - m) * m

        def forward(self, input, label):
            # cos(theta) & phi(theta)
            # torch.nn.functional.linear(input, weight, bias=None)
            # y = x * W^T + b
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            # cos(a+b) = cos(a) * cos(b) - size(a) * sin(b)
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                # torch.where(condition, x, y) --> Tensor
                # 
                # 
                # 
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine-self.m)
            # convert label to one_hot
            # one_hot = torch.zeros(cosine.size(), device='cuda')
            # 将 cos(\theta+m) 更新到 tensor 相应的位置中
            one_hot = torch.zeros(cosine.size(), device='cuda')
            # scatter_(dim, index, src)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            # torch.where(out_i = {x_i if condition_i else y_i})
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s

            return output


# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Par9.5 PyTorch 实现人脸检测与识别
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
if ParX_val == 'Par9.5':
    from PIL import Image
    from face_dect_recong.align.detector import detect_faces
    from face_dect_recong.align.visualization_utils import show_results

    img = Image.open('')

