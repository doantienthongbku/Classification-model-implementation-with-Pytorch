import torch
import torch.nn as nn

class ResBlock_original(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super(ResBlock_original, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.stride = stride
        self.down_sample = down_sample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x.clone()
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.down_sample is not None:
            identity = self.down_sample(identity)
        
        x = x + identity
        x = self.relu(x)
        return x
    

class ResBlock_iden(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
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
        self.down_sample = down_sample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x.clone()
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        if self.down_sample is not None:
            identity = self.down_sample(identity)
        
        x = x + identity
        x = self.relu(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, Block, layer_list, num_classes, img_channels):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        # make some first layers
        self.conv1 =  nn.Conv2d(in_channels=img_channels, out_channels=64,
                                kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layers(Block, layer_list[0], 64 , stride=1)
        self.layer2 = self._make_layers(Block, layer_list[1], 128, stride=2)
        self.layer3 = self._make_layers(Block, layer_list[2], 256, stride=2)
        self.layer4 = self._make_layers(Block, layer_list[3], 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512 * Block.expansion, num_classes) 
        
        self._initial_parametter()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x

    def _make_layers(self, Block, num_blocks, out_channels, stride=1):
        down_sample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels * Block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels  * Block.expansion,
                          kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channels * Block.expansion)
            )
        
        layers.append(Block(self.in_channels, out_channels, stride, down_sample))
        self.in_channels = out_channels  * Block.expansion
        
        for _ in range(num_blocks - 1):
            layers.append(Block(self.in_channels, out_channels))
            
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


def ResNet18(num_classes, channels=3):
    return ResNet(ResBlock_original, [2, 2, 2, 2], num_classes, channels)

def ResNet334(num_classes, channels=3):
    return ResNet(ResBlock_original, [3, 4, 6, 3], num_classes, channels)

def ResNet50(num_classes, channels=3):
    return ResNet(ResBlock_iden, [3, 4, 6, 3], num_classes, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(ResBlock_iden, [3, 4, 23, 3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(ResBlock_iden, [3, 8, 36, 3], num_classes, channels)


def test(device='cpu'):
    net =ResNet18(num_classes=100)
    # print(net)
    # from torchsummary import summary
    # summary(model=net, input_size=(3, 224, 224), device='cpu')
    x = torch.randn(64, 3, 224, 224)     # output: torch.Size([64, 100])
    y = net(x).to(device)
    print(y.shape)
    
# test()