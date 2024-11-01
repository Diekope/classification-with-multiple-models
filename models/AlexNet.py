import torch
import torch.nn as nn
from .BaseModel import BaseModel

# Configuration de l'appareil
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AlexNet(BaseModel):
    def __init__(self, num_classes=9):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))


        # Remplacer la couche entièrement connectée avec une adaptation à  l'entrée
        # Automatiser la détection de la taille de la sortie pour les couches FC
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Pooling adaptatif pour gérer diverses tailles d'entée
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 4096), # Ajustement des dimensions basé sur le global pooling
            nn.ReLU())

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())

        self.fc2 = nn.Linear(4096, num_classes)  # Ajustement ici

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
