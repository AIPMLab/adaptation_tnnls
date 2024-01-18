import torch.nn as nn
from torchvision import models

resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}


def get_backbone(name):
    if "resnet" in name.lower():
        return ResNetBackbone(name)
    elif "alexnet" == name.lower():
        return AlexNetBackbone()
    elif "dann" == name.lower():
        return DaNNBackbone()
    elif "efficient" == name.lower():
        return EfficientNetBackbone()
    elif "res" == name.lower():
        return ResNeStBackbone()
    elif "vit" == name.lower():
        return ViTBackbone()
    elif "densenet" == name.lower():
        return DenseNetBackbone()
    elif "googlenet" == name.lower():
        return GoogleNetBackbone()
    elif 'mnasnet' == name.lower():
        return mnasnet1_0Backbone()
    elif 'vgg' == name.lower():
        return VGGBackbone()
    elif 'shuffle' == name.lower():
        return ShuffleBackbone()


class DaNNBackbone(nn.Module):
    def __init__(self, n_input=224 * 224 * 3, n_hidden=256):
        super(DaNNBackbone, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self._feature_dim = n_hidden

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def output_num(self):
        return self._feature_dim

# convnet without the last layer
class AlexNetBackbone(nn.Module):
    def __init__(self):
        super(AlexNetBackbone, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier" + str(i), model_alexnet.classifier[i])
        self._feature_dim = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self._feature_dim

class ResNetBackbone(nn.Module):
    def __init__(self, network_type):
        super(ResNetBackbone, self).__init__()
        if network_type == 'resnet34':
            resnet = resnet_dict[network_type](weights='ResNet34_Weights.DEFAULT')
        elif network_type == 'resnet50':
            resnet = resnet_dict[ network_type ](weights='ResNet50_Weights.DEFAULT')
        elif network_type == 'resnet101':
            resnet = resnet_dict[ network_type ](weights='ResNet101_Weights.DEFAULT')
        else:
            resnet = resnet_dict[ network_type ](weights='ResNet18_Weights.DEFAULT')
        # print(resnet)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = nn.BatchNorm2d(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self._feature_dim


from efficientnet_pytorch import EfficientNet

class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name='efficientnet-b2', pretrained=True):
        super(EfficientNetBackbone, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name) if pretrained else EfficientNet.from_name(model_name)
        self._feature_dim = self.model._fc.in_features

    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.model._avg_pooling(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self._feature_dim

import torch
import torch.nn as nn

class ResNeStBackbone(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNeStBackbone, self).__init__()
        self.model = resnet_dict['resnet50'](pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 2048)
        self._feature_dim = self.model.fc.in_features

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x

    def output_num(self):
        return self._feature_dim

import torch
import torch.nn as nn
import torchvision.models as models

class DenseNetBackbone(nn.Module):
    def __init__(self, model_name='densenet121', pretrained=True):
        super(DenseNetBackbone, self).__init__()
        # self.model = models.DenseNet.from_pretrained(model_name) if pretrained else models.DenseNet.from_name(model_name)
        self.model = models.densenet121(pretrained=True)
        self._feature_dim = self.model.classifier.in_features

    def forward(self, x):
        features = self.model.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out

    def output_num(self):
        return self._feature_dim


from torchvision.models import googlenet
from torchvision.models import squeezenet1_1

class GoogleNetBackbone(nn.Module):
    def __init__(self):
        super(GoogleNetBackbone, self).__init__()
        self.googlenet = googlenet(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.googlenet.fc = nn.Linear(in_features=1024, out_features=1024)
        self._feature_dim = self.googlenet.fc.out_features

    def forward(self, x):
        # x = self.
        x = self.googlenet(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self._feature_dim

from torchvision.models import mnasnet1_0
from torchvision.models import vgg16

class mnasnet1_0Backbone(nn.Module):
    def __init__(self):
        super(mnasnet1_0Backbone, self).__init__()
        self.mnasnet1_0 = mnasnet1_0(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # print(self.mnasnet1_0)
        # self.googlenet.fc = nn.Linear(in_features=1024, out_features=1024)
        self._feature_dim = self.mnasnet1_0.classifier[1].out_features

    def forward(self, x):
        # x = self.
        x = self.mnasnet1_0(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self._feature_dim

class VGGBackbone(nn.Module):
    def __init__(self):
        super(VGGBackbone, self).__init__()
        self.VGG = vgg16(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # print(self.VGG)
        # self.googlenet.fc = nn.Linear(in_features=1024, out_features=1024)
        self._feature_dim = self.VGG.classifier[6].out_features

    def forward(self, x):
        # x = self.
        x = self.VGG(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self._feature_dim

from torchvision.models import shufflenet_v2_x1_0

class ShuffleBackbone(nn.Module):
    def __init__(self):
        super(ShuffleBackbone, self).__init__()
        self.VGG = shufflenet_v2_x1_0(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # print(self.VGG)
        # self.googlenet.fc = nn.Linear(in_features=1024, out_features=1024)
        self._feature_dim = self.VGG.fc.out_features

    def forward(self, x):
        # x = self.
        x = self.VGG(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self._feature_dim