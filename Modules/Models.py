# -*- coding: utf-8 -*-
"""
Created on Wed Mar  13 12:47:47 2019

@author: Albert
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Models(nn.Module):
    def __init__(self, numberOfClasses, pretrained, model):
        super(Models, self).__init__()

        self.numberOfClasses = numberOfClasses
        self.pretrained = pretrained
        self.modelName = model
        self.net = self.modelArchitecture()
        self.numOfInputFeaturesInFullyConnectedLayer = self.net.classifier.in_features
        self.net.classifier = nn.Linear(self.numOfInputFeaturesInFullyConnectedLayer, self.numberOfClasses)
        self.net.classifier = nn.Linear(self.numOfInputFeaturesInFullyConnectedLayer, 1000)
        self.net.fc = nn.Linear(self.numOfInputFeaturesInFullyConnectedLayer, self.numberOfClasses)

    def forward(self, x):
        x = self.net(x)
        return x

    def modelArchitecture(self):
        return models.densenet121(pretrained = self.pretrained)
