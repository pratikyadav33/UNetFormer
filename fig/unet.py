import torch
import torch.nn as nn
import torchvision.models as models

class FTUnetformer(nn.Module):
    def __init__(self, num_classes=16):
        super(FTUnetformer, self).__init__()

        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)

        # Remove fully connected layer and average pooling
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # Upsampling path
        self.upconv1 = self._conv_block(512, 256)
        self.upconv2 = self._conv_block(256, 128)
        self.upconv3 = self._conv_block(128, 64)
        self.upconv4 = self._conv_block(64, 64)

        # Output convolutional layer
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder (backbone)
        x1 = self.backbone[0](x)
        x2 = self.backbone[1](x1)
        x3 = self.backbone[2](x2)
        x4 = self.backbone[3](x3)

        # Decoder (upsampling path)
        x = self.upconv1(x4)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.upconv2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.upconv3(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.upconv4(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Output convolutional layer
        x = self.out_conv(x)

        return x

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

