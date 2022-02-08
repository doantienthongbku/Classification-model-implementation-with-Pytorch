import torch
import torch.nn as nn

class ResBlock_original(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock_original, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.stride = stride
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x.clone()
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if (identity.shape != x.shape):
            identity = self.down_sample(identity)
        
        x = x + identity
        x = self.relu(x)
        return x
    

class ResBlock_iden(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock_iden, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels * self.expansion)
        
        self.stride = stride
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x.clone()
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        if (identity.shape != x.shape):
            identity = self.down_sample(identity)
        
        x = x + identity
        x = self.relu(x)
        return x
    
class ResNet_CIFAR(nn.Module):
    def __init__(self, Block, n, classes):
        super(ResNet_CIFAR, self).__init__()
        
        self.n = n
        self.Block = Block
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU(inplace=True)
        
        self.hidden_layers = self._make_layers(n)
        
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(in_features=256, out_features=classes)
        
        self._initial_parametter()
        
        
    def forward(self, x):
        x = self.relu(self.bn0(self.conv0(x)))
        x = self.hidden_layers(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
    
    def _make_layers(self, n):
        layers = []
        
        in_channels = 16
        out_channels = 16
        stride = 1
        for i in range(3):
            for j in range(self.n):
                if i > 0 and j == 0:
                    in_channels = out_channels
                    out_channels *= 2
                    stride = 2
                
                layers.append(self.Block(in_channels, out_channels, stride))
                
                stride = 1
                in_channels = out_channels
                
        return nn.Sequential(*layers)
    
    def _initial_parametter(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    
def ResNet20_CIFAR(classes):
    return ResNet_CIFAR(ResBlock_original, 3, classes)

def ResNet32_CIFAR(classes):
    return ResNet_CIFAR(ResBlock_original, 5, classes)

def ResNet44_CIFAR(classes):
    return ResNet_CIFAR(ResBlock_original, 7, classes)

def ResNet56_CIFAR(classes):
    return ResNet_CIFAR(ResBlock_original, 9, classes)

def test(device='cpu'):
    net = ResNet56_CIFAR(100)
    from torchsummary import summary
    summary(model=net, input_size=(3, 224, 224), device='cpu')
    x = torch.randn(64, 3, 32, 32)
    y = net(x).to(device)
    print(y.shape)
    
test()