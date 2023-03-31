import torch
import torch.nn as nn

class ConvolutionBlock(nn.Module):

    def __init__(self, in_channels, out_channels): # h, w -> h/2, w/2 | # of kernel -> 2 * # of kernel
        self.in_channels = in_channels
        self.out_channels = out_channels

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels), # 출력의 크기보다는 output chanel의 숫자를 넣어줘야 함
            nn.Conv2d(out_channels, out_channels, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )


    def forward(self, x):
        # |x| = (batch_size, in_channels, h, w)

        y = self.layers(x)
        # |y| = (batch_size, out_channels, h, w)

        return y

class ConvolutionalClassifier(nn.Module):

    def __init__(self, output_size):
        self.output_size = output_size

        super().__init__()

        self.blocks = nn.Sequential( # |x| = (n, 1, 28, 28)
            ConvolutionBlock(1, 32), # (n, 32, 14, 14)
            ConvolutionBlock(32, 64), # (n, 64, 7, 7)
            ConvolutionBlock(64, 128), # (n, 128, 4, 4)
            ConvolutionBlock(128, 256), # (n, 256, 2, 2)
            ConvolutionBlock(256, 512), # (n, 512, 1, 1)
        )

        self.layers = nn.Sequential(
            nn.Linear(512, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x):
        assert x.dim() > 2 # flatten을 실수로 하지 않을 경우를 대비해서 assert를 걸어 놓음 fc 인 경우, (bs, 784) dim = 2

        if x.dim() == 3: # grayscale인 경우 - tensor 모양 변경
            # |x| = (batch_size, h, w)
            x = x.view(-1, 1, x.size(-2), x.size(-1))
            # |x| = (batch_size, 1, h, w)

        z = self.blocks(x)
        # |z| = (batch_size, 512, 1, 1)

        y = self.layers(z.squeeze())
        # |y| = (batch_size, output_size)

        return y