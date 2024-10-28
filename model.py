import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()

        # Layer 1
        self.layer1 = nn.Sequential(
            CNNBlock(3, 32, kernel_size=3, padding=1),
            CNNBlock(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layer 2
        self.layer2 = nn.Sequential(
            CNNBlock(32, 64, kernel_size=3, padding=1),
            CNNBlock(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layer 3
        self.layer3 = nn.Sequential(
            CNNBlock(64, 128, kernel_size=3, padding=1),
            CNNBlock(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layer 4
        self.layer4 = nn.Sequential(
            CNNBlock(128, 256, kernel_size=3, padding=1),
            CNNBlock(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layer 5
        self.layer5 = nn.Sequential(
            CNNBlock(256, 512, kernel_size=3, padding=1),
            CNNBlock(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 16 * 16, 1024),  # Assuming input images of size 524x524
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Flatten the output before the fully connected layer
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        return x


# -----------------------------------
# Création modèle Google-Net
# -----------------------------------
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(InceptionModule, self).__init__()

        # 1x1 convolution branch
        self.branch1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        # 3x3 convolution branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1)
        )

        # 5x5 convolution branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2)
        )

        # Max pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class GoogleNet(nn.Module):
    def __init__(self, num_classes=6):
        super(GoogleNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer3 = nn.Sequential(
            InceptionModule(192, 64, 96, 128, 16, 32, 32),
            InceptionModule(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer4 = nn.Sequential(
            InceptionModule(480, 192, 96, 208, 16, 48, 64),
            InceptionModule(512, 160, 112, 224, 24, 64, 64),
            InceptionModule(512, 128, 128, 256, 24, 64, 64),
            InceptionModule(512, 112, 144, 288, 32, 64, 64),
            InceptionModule(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer5 = nn.Sequential(
            InceptionModule(832, 384, 160, 384, 32, 128, 128),  # Entrée = 832 canaux, Sortie = 1024 canaux
            InceptionModule(1024, 384, 192, 384, 48, 128, 128),  # Entrée = 1024 canaux, Sortie = 1024 canaux
            nn.AvgPool2d(kernel_size=7))

        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        # Flatten the output before fully connected layer
        out = torch.flatten(out, 1)

        out = self.fc(out)
        return out
