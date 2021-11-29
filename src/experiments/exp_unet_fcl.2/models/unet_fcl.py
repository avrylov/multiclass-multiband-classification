
import torch
from torchsummary import summary

from .unet import UNet


class FCnn(torch.nn.Module):

    def __init__(self):
        super(FCnn, self).__init__()

        self.band_out_unet = UNet(n_channels=13, n_classes=13)
        self.pool = torch.nn.MaxPool2d(kernel_size=(4, 4))

        self.fc1 = torch.nn.Linear((13 * 16 * 16), 4096)
        self.fc_act1 = torch.nn.ReLU()
        self.fc_drop1 = torch.nn.Dropout(p=0.25)

        self.fc2 = torch.nn.Linear(4096, 10)

    def forward(self, bands):
        bands_u = self.band_out_unet(bands)
        bands = self.pool(bands_u)

        bands = bands.view(bands.size(0), bands.size(1) * bands.size(2) * bands.size(3))

        bands = self.fc1(bands)
        bands = self.fc_act1(bands)
        bands = self.fc_drop1(bands)

        bands = self.fc2(bands)

        return bands


if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = FCnn().to(device)

    print(summary(model, (13, 64, 64)))

