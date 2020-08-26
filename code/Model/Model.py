# -*- coding: utf-8 -*-
from torch import nn
import torch
import torchvision.models as Models

class TextfeatureNet(nn.Module):
    def __init__(self, neure_num):
        super(TextfeatureNet, self).__init__()
        self.mlp = make_layers(neure_num[:-1])
        self.feature = nn.Linear(neure_num[-2], neure_num[-1])

    def forward(self, x):
        temp_x = self.mlp(x)
        x = self.feature(temp_x)
        return x

class PredictNet(nn.Module):
    def __init__(self, neure_num):
        super(PredictNet, self).__init__()
        self.mlp = make_predict_layers(neure_num)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.mlp(x)
        y = self.sigmoid(y)
        return y


class AttentionNet(nn.Module):
    def __init__(self, neure_num):
        super(AttentionNet, self).__init__()
        self.mlp = make_layers(neure_num[:-1])
        self.attention = nn.Linear(neure_num[-2], neure_num[-1])

    def forward(self, x):
        temp_x = self.mlp(x)
        y = self.attention(temp_x)
        return y

class ImgNet(nn.Module):
    def __init__(self):
        super(ImgNet, self).__init__()
        self.feature = Models.resnet18(pretrained = True)
        self.feature = nn.Sequential(*list(self.feature.children())[:-1])
        self.fc1 = nn.Sequential(       
            nn.Linear(512, 128)
        )

    def forward(self, x):
        N = x.size()[0]
        x = self.feature(x.view(N, 3, 256, 256))
        x = x.view(N, 512)
        x = self.fc1(x)
        return x

def make_layers(cfg):
    layers = []
    n = len(cfg)
    input_dim = cfg[0]
    for i in range(1, n):
        output_dim = cfg[i]
        layers += [nn.Linear(input_dim, output_dim), nn.ReLU(inplace = True)]
        input_dim = output_dim
    return nn.Sequential(*layers)

def make_predict_layers(cfg):
    layers = []
    n = len(cfg)
    input_dim = cfg[0]
    for i in range(1, n):
        output_dim = cfg[i]
        layers += [nn.Linear(input_dim, output_dim)]
        input_dim = output_dim
    return nn.Sequential(*layers)

def generate_model(Textfeatureparam, Imgpredictparam, Textpredictparam, Attentionparam, Predictparam):
    Textfeaturemodel = TextfeatureNet(Textfeatureparam)
    Imgpredictmodel = PredictNet(Imgpredictparam)
    Textpredictmodel = PredictNet(Textpredictparam)
    Predictmodel = PredictNet(Predictparam)
    Imgmodel = ImgNet()
    Attentionmodel = AttentionNet(Attentionparam)

    return Textfeaturemodel, Imgpredictmodel, Textpredictmodel, Imgmodel, Attentionmodel, Predictmodel
