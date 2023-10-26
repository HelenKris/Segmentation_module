from torchvision.models import vgg16_bn
import torch.nn as nn
import torch


# Для того, чтобы облегчить нам работу с построением архитектуры, создадим два отдельных блока,
# один из которых будет отвечать за операции
# в левой части U-Net архитектуры (то есть сжатие изображение),
# а другой будет отвечать за операции в правой части архитектуры (то есть апсемплинг):

# сверточный блок для левой части архитектуры

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def convolve(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True)
    )

    # апсемплинг блок для правой части архитектуры

def up_convolve(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2),
        nn.ReLU(inplace = True)
    )

# Теперь определим всю архитектуру для U-Net:

class UNet(nn.Module):
    def __init__(self, pretrained=True, out_channels=4):
        super().__init__()
        # Будем использовать модель VGG-16 в качестве энкодера
        self.encoder = vgg16_bn(pretrained=pretrained).features
        # Разделим VGG поблочно на отдельные блоки чтобы было удобно их модифицировать и обращаться к ним
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])

        # Боттлнек нашей архитектуры (самый глубокий уровень сети -> смотри архитектуру U-Net на картинке выше)
        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck = convolve(512, 1024)

        # Блоки декодера с "транспонированными свертками" и сверточными блоками
        self.up_conv6 = up_convolve(1024, 512)
        self.conv6 = convolve(512 + 512, 512)
        self.up_conv7 = up_convolve(512, 256)
        self.conv7 = convolve(256 + 512, 256)
        self.up_conv8 = up_convolve(256, 128)
        self.conv8 = convolve(128 + 256, 128)
        self.up_conv9 = up_convolve(128, 64)
        self.conv9 = convolve(64 + 128, 64)
        self.up_conv10 = up_convolve(64, 32)
        self.conv10 = convolve(32 + 64, 32)
        # Финальный сверточный слой для приведения к нужному числу каналов
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Проход через блоки энкодера
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        # Проход через блок боттлнека
        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        # Проход через блоки декодера с использованием "skip connections"
        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)

        # Проход через финальный свёрточный слой
        x = self.conv11(x)

        return x
