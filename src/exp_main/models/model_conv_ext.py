import torch
from torchsummary import summary


class ConvNet(torch.nn.Module):
    '''
    simple convolution architecture for 7 multiband geo spatial data
    '''
    def __init__(self):
        super(ConvNet, self).__init__()

        # 1st stage
        self.conv1_1 = torch.nn.Conv2d(
            in_channels=13, out_channels=32, kernel_size=(3, 3), padding='same')
        self.bn1_1 = torch.nn.BatchNorm2d(32)
        self.act1_1 = torch.nn.ELU()

        self.conv1_2 = torch.nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=(3, 3))
        self.bn1_2 = torch.nn.BatchNorm2d(32)
        self.act1_2 = torch.nn.ELU()

        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop1 = torch.nn.Dropout(p=0.25)

        # 2nd stage
        self.conv2_1 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding='same')
        self.bn2_1 = torch.nn.BatchNorm2d(64)
        self.act2_1 = torch.nn.ReLU()

        self.conv2_2 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3))
        self.bn2_2 = torch.nn.BatchNorm2d(64)
        self.act2_2 = torch.nn.ReLU()

        self.pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop2 = torch.nn.Dropout(p=0.25)

        # 3rd stage
        self.conv3_1 = torch.nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same')
        self.bn3_1 = torch.nn.BatchNorm2d(128)
        self.act3_1 = torch.nn.ReLU()

        self.pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop3 = torch.nn.Dropout(p=0.25)

        self.fc1 = torch.nn.Linear((128 * 2 * 2), 128)
        self.fc_act1 = torch.nn.ReLU()

        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, bands):
        # 1st stage
        bands = self.conv1_1(bands)
        bands = self.bn1_1(bands)
        bands = self.act1_1(bands)

        bands = self.conv1_2(bands)
        bands = self.bn1_2(bands)
        bands = self.act1_2(bands)

        bands = self.pool1(bands)
        bands = self.drop1(bands)

        # 2nd stage
        bands = self.conv2_1(bands)
        bands = self.bn2_1(bands)
        bands = self.act2_1(bands)

        bands = self.conv2_2(bands)
        bands = self.bn2_2(bands)
        bands = self.act2_2(bands)

        bands = self.pool2(bands)
        bands = self.drop2(bands)

        # 3rd stage
        bands = self.conv3_1(bands)
        bands = self.bn3_1(bands)
        bands = self.act3_1(bands)

        bands = self.pool3(bands)
        bands = self.drop3(bands)

        bands = bands.view(bands.size(0), bands.size(1) * bands.size(2) * bands.size(3))
        #
        bands = self.fc1(bands)
        bands = self.fc_act1(bands)
        #
        # bands = self.fc2(bands)
        # bands = self.fc_act2(bands)
        #
        bands = self.fc3(bands)

        return bands


device = torch.device("cuda:0")
model = ConvNet().to(device)

print(summary(model, (13, 32, 32)))

