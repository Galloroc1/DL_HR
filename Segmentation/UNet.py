import torch.nn as nn
import torch


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DoubleConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same'))

    def forward(self, x):
        x = self.double_conv(x)
        return x


class OutConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(OutConvBlock, self).__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same'))

    def forward(self, x):
        x = self.out_conv(x)
        return x


"""
encoder block:max_pool->conv2->conv2
"""

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


"""
decoder: ConvTranspose2d->conv->conv
"""
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3)

    def forward(self, x, pad_x):
        x = self.up(x)
        # cat
        concatenated_tensor = torch.cat([x, pad_x], dim=1)
        # conv
        out = self.conv(concatenated_tensor)
        return out


class Unet(nn.Module):
    def __init__(self, class_num, in_channels=3, base_c=64):
        super(Unet, self).__init__()
        self.in_conv = DoubleConvBlock(in_channels=in_channels, out_channels=base_c)
        self.down1 = DownBlock(in_channels=base_c, out_channels=base_c * 2)
        self.down2 = DownBlock(in_channels=base_c * 2, out_channels=base_c * 4)
        self.down3 = DownBlock(in_channels=base_c * 4, out_channels=base_c * 8)
        self.down4 = DownBlock(in_channels=base_c * 8, out_channels=base_c * 16)

        self.up1 = UpBlock(in_channels=base_c * 16, out_channels=base_c * 8)
        self.up2 = UpBlock(in_channels=base_c * 8, out_channels=base_c * 4)
        self.up3 = UpBlock(in_channels=base_c * 4, out_channels=base_c * 2)
        self.up4 = UpBlock(in_channels=base_c * 2, out_channels=base_c)

        self.out_conv = OutConvBlock(in_channels=base_c, out_channels=class_num, kernel_size=1)

    def forward(self, x):
        # encoder
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        # decoder
        u1 = self.up1(x4, x3)
        u2 = self.up2(u1, x2)
        u3 = self.up3(u2, x1)
        u4 = self.up4(u3, x0)

        # out_layer
        out = self.out_conv(u4)
        return out


# if __name__ == '__main__':
#     class_num = 5
#     x = torch.randn(8, 3, 512, 512)
#     model = Unet(class_num, in_channels=x.shape[1])
#     out = model(x)
#     print(out.shape)
