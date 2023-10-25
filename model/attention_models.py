#%%

'''
Refs:
1. Resnet_cbam: https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
2. SE-ResNext50 but with attention : https://www.kaggle.com/code/debarshichanda/seresnext50-but-with-attention
--> 여기 나오는 SAM optimizer 써보기

3. BoTnet: 
--> "By just replacing the spatial convolutions with global self-attention in the final three bottleneck blocks"
--> "BoTNet, we also point out how ResNet bottleneck blocks with self-attention can be viewed as Transformer blocks."
1) Official: https://github.com/lucidrains/bottleneck-transformer-pytorch/blob/main/bottleneck_transformer_pytorch/bottleneck_transformer_pytorch.py
2) Non-official: https://github.com/MartinGer/Bottleneck-Transformers-for-Visual-Recognition/blob/master/BoTNet_Layer.py

4. SACNN: https://github.com/guoyii/SACNN/blob/master/model_function.py 

5. ViT in 3D from BraTS-Radiogenomics challenge: 
1) https://www.kaggle.com/code/super13579/vit-vision-transformer-3d-with-one-mri-type
--> "I use @ROLAND LUETHY great notebook, and change the efficient3D model to VIT 3D model, just want to try Vit on 3D"
2) https://www.kaggle.com/code/rluethy/efficientnet3d-with-one-mri-type/notebook 
--> "Use models with only one MRI type, then ensemble the 4 models"

6. ViT-V-Net: IXI, ADNI, OASIS, ABIDE 등 다른 brain MRI open dataset 으로 registration 했던 연구
https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch
'''

'''
(20221017 오전 완료) 참고: 
1) 현재 전처리 2mm resize로 SNUH, severance, UPenn dataset 모두 마쳐서 resized_BraTS 디렉토리 안에
각각 t1ce_resized.nii.gz 이렇게 들어있음. 
2) 문제는 아직, UPenn의 seg.nii.gz 에서 ET 4->2, ED 2->1 로 고쳐야 하는데, 
3) Necrosis가 1인지 0인지 확인해보고 hd_glio에 맞게 다시 label 값 고쳤고, seg_resized.nii.gz로 저장했고,
4) 원래의 UPenn label 따라 한 건 brats_seg_resized.nii.gz 로 이름 바꿔 저장함.

20221016 저녁 앞으로 할 일 결론:
1. 총 3개의 코드를 조합한다.
1) 전체 골격 코드 (https://www.kaggle.com/code/debarshichanda/seresnext50-but-with-attention)
2) 그 안의 BottleBlock 부분 소스코드 (https://github.com/lucidrains/bottleneck-transformer-pytorch/blob/main/bottleneck_transformer_pytorch/bottleneck_transformer_pytorch.py)
3) 그 안의 Attention 부분 바꿀 소스코드 (https://github.com/guoyii/SACNN/blob/master/model_function.py)

2. /mnt/hdd2/kschoi/GBL/code/attention_practice.py 에다가, 
1)의 골격 코드 그대로 쓰되, 2)에 나오는 BottleBlock 부분 구현을 바꾸고, 구체적으로 3)의 SACNN 에서 나온 attention 
코드로 바꿔서 차원을 맞춰준다.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['se_resnext50', 'resnet50_cbam']
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed3d.pth',
# }


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
           
        self.fc = nn.Sequential(nn.Conv3d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv3d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SElayer(nn.Module):
    def __init__(self, inplanes, reduction=16):
        super(SElayer,self).__init__()
        self.globalAvgpool = nn.AdaptiveAvgPool3d(1)#Squeeze操作
        self.fc1 = nn.Conv3d(inplanes, inplanes // reduction, kernel_size=1, stride=1)
        self.fc2 = nn.Conv3d(inplanes // reduction, inplanes, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        begin_input = x
        x = self.globalAvgpool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x * begin_input

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(4, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1) # 원래 ((1, 1)) 였는데 3차원이 되면서 1로 고쳐야 함
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # print(f'before avgpool:{x.shape}') # torch.Size([1, 2048, 3, 3, 3])
        x = self.avgpool(x)
        # print(f'after avgpool:{x.shape}') # torch.Size([1, 2048, 1, 1, 1])
        x = x.view(x.size(0), -1)
        print(f'before fc:{x.shape}') # torch.Size([1, 2048, 1, 1, 1])
        x = self.fc(x)
        print(f'after fc:{x.shape}') # torch.Size([1, 2048, 1, 1, 1])

        return x

class SEResNext(nn.Module):
    
    def __init__(self, block, layers, num_classes, cardinality = 32):
        super(SEResNext, self).__init__()
        self.cardinality = cardinality
        self.inplanes = 64
        
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        
        #参数初始化，待懂
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def _make_layer(self, block, outplanes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != outplanes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, outplanes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(outplanes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, outplanes, self.cardinality, stride, downsample))
        self.inplanes = outplanes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, outplanes, self.cardinality))


        return nn.Sequential(*layers)
        
    #def forward(self, x):
    #    x = self.conv1(x)
    #    x = self.bn1(x)
    #    x = self.relu(x)
    #    x = self.maxpool(x)
    
    #    x = self.layer1(x)
    #    x = self.layer2(x)
    #    x = self.layer3(x)
    #    x = self.layer4(x)
    
    #    x = self.avgpool(x)
    #    x = x.view(x.size(0), -1)
            
    #    x = self.fc(x)
            
    #    fc_graph = torch.nn.Linear(x.in_features, 168)
    #    fc_vowel = torch.nn.Linear(x.in_features, 11)
    #    fc_conso = torch.nn.Linear(x.in_features, 7)
            
    #    return fc_graph, fc_vowel, fc_conso
        
        
def se_resnext50(**kwargs):
    
    model = SEResNext(Bottleneck, [3,4,6,3], **kwargs)
    return model

def resnet50_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model

# %%
