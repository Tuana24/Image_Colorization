import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(inplace=True)
        )


    def forward(self, x):

        return self.block(x)



#colorization network
class ColorizationNet(nn.Module):

    def __init__(self):

        super().__init__()
        # ENCODER

        self.enc1 = ConvBlock(
            1,
            32
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(
            32,
            64
        )

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(
            64,
            128
        )

        self.pool3 = nn.MaxPool2d(2)

##bottleneck
        self.bottleneck = ConvBlock(
            128,
            256
        )

##decoder
        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = ConvBlock(
            256,
            128
        )
        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )
        self.dec2 = ConvBlock(
            128,
            64
        )
        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )
        self.dec1 = ConvBlock(
            64,
            32
        )

##output
        self.output = nn.Conv2d(
            32,
            2,
            kernel_size=1
        )

        self.tanh = nn.Tanh()

    def forward(self, x):

        # Encoder

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)


        # Bottleneck

        b = self.bottleneck(p3)


        # Decoder

        d3 = self.up3(b)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)


        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)


        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)


        # Output

        output = self.output(d1)

        output = self.tanh(output)

        return output



if __name__ == "__main__":

    model = ColorizationNet()

    test_input = torch.randn(
        1,
        1,
        96,
        96
    )

    output = model(test_input)

    print(
        "Input shape:",
        test_input.shape
    )

    print(
        "Output shape:",
        output.shape
    )