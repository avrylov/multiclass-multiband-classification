import torch

from model_resnet import ResNet
from model_conv import ConvNet


class SiamNN(torch.nn.Module):
    def __init__(self):
        super(SiamNN, self).__init__()

        # out from ConvNet for 7 multiband geospatial data
        self.conv_band_out = ConvNet()

        # out from ConvNet for 13 multiband geospatial data
        self.res_band_out = ResNet()

        # fc block 1
        self.fc1 = torch.nn.Linear((64 * 5 * 5) + 2, 128)  # out from ConvNet + 2 features
        self.fc_act1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(p=0.25)

        # fc block 2
        self.fc2 = torch.nn.Linear(128 + 128, 16)  # out from fc block 1
        self.fc_act2 = torch.nn.ReLU()

        # fc block 3
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, bands_7, bands_13, features):

        # out from ConvNet for 7 multiband geospatial data
        bands_7 = self.conv_band_out(bands_7)

        # out from ConvNet for 13 multiband geospatial data
        bands_13 = self.res_band_out(bands_13)

        # fc block 1
        bands_7_f = torch.cat((bands_7, features), dim=1)  # concat band_7 and 2 features
        bands_7_f = self.fc1(bands_7_f)
        bands_7_f = self.fc_act1(bands_7_f)
        bands_7_f = self.drop1(bands_7_f)

        # fc block 2
        out = torch.cat((bands_7_f, bands_13), dim=1)  # concat bands_7_f and bands_13
        out = self.fc2(out)
        out = self.fc_act2(out)

        # fc block 3
        out = self.fc3(out)

        return out

