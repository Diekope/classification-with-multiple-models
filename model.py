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
