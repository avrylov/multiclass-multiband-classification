import torch
import torch.nn
from torchsummary import summary


class ResIdentity(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()

        # first block
        self.block1_conv = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=in_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding='valid')
        self.block1_bn = torch.nn.BatchNorm2d(in_channels)
        self.block1_act = torch.nn.ReLU()

        # second block
        self.block2_conv = torch.nn.Conv2d(
            in_channels,
            out_channels=in_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding='same')
        self.block2_bn = torch.nn.BatchNorm2d(in_channels)
        self.block2_act = torch.nn.ReLU()

        # third block
        self.block3_conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding='valid')
        self.block3_bn = torch.nn.BatchNorm2d(out_channels)

        # add
        self.block_add_act = torch.nn.ReLU()

    def forward(self, x):
        x_skip = x
        # first block
        x = self.block1_conv(x)
        x = self.block1_bn(x)
        x = self.block1_act(x)

        # second block
        x = self.block2_conv(x)
        x = self.block2_bn(x)
        x = self.block2_act(x)

        # third block
        x = self.block3_conv(x)
        x = self.block3_bn(x)

        # add
        x = torch.add(x, x_skip)
        x = self.block_add_act(x)

        return x


class ResConv(torch.nn.Module):
    def __init__(self, in_channels, f1, f2, stride):
        super().__init__()

        # first block
        self.block1_conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=f1,
            kernel_size=(1, 1),
            stride=stride, padding='valid')
        self.block1_bn = torch.nn.BatchNorm2d(f1)
        self.block1_act = torch.nn.ReLU()

        # second block
        self.block2_conv = torch.nn.Conv2d(
            in_channels=f1,
            out_channels=f1,
            kernel_size=(3, 3),
            stride=(1, 1), padding='same')
        self.block2_bn = torch.nn.BatchNorm2d(f1)
        self.block2_act = torch.nn.ReLU()

        # third block
        self.block3_conv = torch.nn.Conv2d(
            in_channels=f1,
            out_channels=f2,
            kernel_size=(1, 1),
            stride=(1, 1), padding='valid')
        self.block3_bn = torch.nn.BatchNorm2d(f2)

        # shortcut
        self.block_sh_conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=f2,
            kernel_size=(1, 1),
            stride=(1, 1), padding='valid')
        self.block_sh_bn = torch.nn.BatchNorm2d(f2)

        # add
        self.block_add_act = torch.nn.ReLU()

    def forward(self, x):
        x_skip = x
        # first block
        x = self.block1_conv(x)
        x = self.block1_bn(x)
        x = self.block1_act(x)

        # second block
        x = self.block2_conv(x)
        x = self.block2_bn(x)
        x = self.block2_act(x)

        # third block
        x = self.block3_conv(x)
        x = self.block3_bn(x)

        # shortcut
        x_skip = self.block_sh_conv(x_skip)
        x_skip = self.block_sh_bn(x_skip)

        # add
        x = torch.add(x, x_skip)
        x = self.block_add_act(x)

        return x


class ResNet(torch.nn.Module):
    '''
    Resnet implementation for 13 multiband geo spatial data
    '''
    def __init__(self):
        super(ResNet, self).__init__()

        self.zero_padding = torch.nn.ZeroPad2d(3)

        # 1st stage
        self.conv1 = torch.nn.Conv2d(
            in_channels=13, out_channels=64, kernel_size=(7, 7), stride=(2, 2))
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.act1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        # 2nd stage
        self._res_conv_bl2 = ResConv(in_channels=64, f1=64, f2=256, stride=1)
        self._res_identity_bl2 = ResIdentity(64, 256)

        # 3rd stage
        self._res_conv_bl3 = ResConv(in_channels=256, f1=128, f2=512, stride=1)
        self._res_identity_bl3 = ResIdentity(128, 512)

        # 4th stage
        self._res_conv_bl4 = ResConv(in_channels=512, f1=256, f2=1024, stride=1)
        self._res_identity_bl4 = ResIdentity(256, 1024)

        # 5th stage
        self.pool_avg = torch.nn.MaxPool2d(kernel_size=2)

        self.fc1 = torch.nn.Linear((512 * 7 * 7), 2048)
        self.fc_act1 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(2048, 512)
        self.fc_act2 = torch.nn.ReLU()

        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, bands):
        bands = self.zero_padding(bands)

        # 1st stage
        bands = self.conv1(bands)
        bands = self.bn1(bands)
        bands = self.act1(bands)
        bands = self.pool1(bands)

        # 2nd stage
        bands = self._res_conv_bl2(bands)
        bands = self._res_identity_bl2(bands)
        bands = self._res_identity_bl2(bands)

        # 3rd stage
        bands = self._res_conv_bl3(bands)
        bands = self._res_identity_bl3(bands)
        bands = self._res_identity_bl3(bands)
        bands = self._res_identity_bl3(bands)

        # 4th stage
        bands = self.pool_avg(bands)

        # 5th stage
        # convert to a vector from avg pooling
        bands = bands.view(bands.size(0), bands.size(1) * bands.size(2) * bands.size(3))

        # vector to fc layers
        bands = self.fc1(bands)
        bands = self.fc_act1(bands)

        # vector to fc layers
        bands = self.fc2(bands)
        bands = self.fc_act2(bands)

        bands = self.fc3(bands)

        return bands


device = torch.device("cuda:0")
model = ResNet().to(device)

print(summary(model, (13, 64, 64)))
