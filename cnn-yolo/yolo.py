import torch.nn as nn
import torch.nn.functional as F
import torch
from pathlib import Path
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, x):
        return self.block(x)

class YOLOv1Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # 448 × 448 × 3
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
            # 224 × 224 × 64

            nn.MaxPool2d(kernel_size=2, stride=2),
            # 112 × 112 × 64

            ConvBlock(64, 192, kernel_size=3, padding=1),
            # 112 × 112 × 192

            nn.MaxPool2d(kernel_size=2, stride=2),
            # 56 × 56 × 192

            ConvBlock(192, 128, kernel_size=1),
            ConvBlock(128, 256, kernel_size=3, padding=1),
            ConvBlock(256, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            # 56 × 56 × 512

            nn.MaxPool2d(kernel_size=2, stride=2),
            # 28 × 28 × 512

            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),
            ConvBlock(512, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3, padding=1),

            ConvBlock(512, 512, 1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),

            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, kernel_size=3, padding=1),

            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1),

            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
        )

        self.prediction = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(4096, )
        )    

    def forward(self, x):
        features = self.features(x)
        pred = self.prediction(features)
        return pred


sample_image = Image.open("cnn-yolo/coco128/images/train2017/000000000009.jpg")
sample_image = sample_image.resize(size=(448, 448))

image_tensor = torch.Tensor(np.array(sample_image)).permute(2, 0, 1)
image_tensor = image_tensor.unsqueeze(dim=0)

# run model
model = YOLOv1Model()

output : torch.Tensor = model(image_tensor)
output = output.reshape(1, 7, 7, 90)

print(output.shape)