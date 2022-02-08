import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchsummary import summary
import torchvision.models as models


class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1, expansion=4):
        super(block, self).__init__()
        hidden = in_channels * expansion
        
        # pw
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=hidden)
        
        # dw
        self.conv2 = nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=7,
                               stride=1, padding=3, groups=hidden, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=hidden)
        
        # pw
        self.conv3 = nn.Conv2d(in_channels=hidden, out_channels=in_channels,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=in_channels)
        
        # conv
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=out_channels)
        
        self.identity_downsample = identity_downsample
        self.mish = nn.Mish(inplace=True)
        
        
    def forward(self, x):
        identity = x
        
        # pw
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.mish(x)
        
        # dw - linear
        x = self.conv2(x)
        x = self.bn2(x)
        
        # pw
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.mish(x)
        
        # conv
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.mish(x)        
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
            
        x += identity
        x = self.mish(x)
        return x
    
    
class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=64,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.mish = nn.Mish(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # resnet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes)
        
        self._initial_parametter()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.mish(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
    
    def _initial_parametter(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def _make_layer(self, block, num_residual_block, out_channels, stride):
        identity_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride, expansion=self.expansion))
        self.in_channels = out_channels
        
        for _ in range(num_residual_block - 1):
            layers.append(block(self.in_channels, out_channels, expansion=self.expansion))
            
        return nn.Sequential(*layers)




def ResNet18(img_channels=3, num_classes=1000):
    return ResNet(block, [2, 2, 2, 2], img_channels, num_classes)


def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)


def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(block, [3, 4, 23, 3], img_channels, num_classes)


def ResNet152(img_channels=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channels, num_classes)


def test(device='cpu'):
    net = ResNet18(num_classes=10)
    print(net)
    summary(model=net, input_size=(3, 32, 32), device='cpu')
    x = torch.randn(1, 3, 224, 224)
    y = net(x).to(device)
    print(y.shape)

# test()
