import torch
import torch.nn as nn
import math

def conv_1x1_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=1, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU6(inplace=True)
    )
    
def conv_3x3_bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU6(inplace=True)
    )

def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = (stride == 1) and (in_channels == out_channels)
        
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                          kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU6(inplace=True),
                
                # pw-lr
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=hidden_dim),
                nn.ReLU6(inplace=True),
                
                # dw
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim,
                          kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(num_features=hidden_dim),
                nn.ReLU6(inplace=True),
                
                # pw-lr
                nn.Conv2d(in_channels=hidden_dim, out_channels=out_channels,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
            
    def forward(self, x):
        if self.use_res_connect:
            return self.conv(x) + x
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, img_size=224, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channels = 32
        last_channels = 1028
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        assert img_size % 32 == 0
        
        # build first layer
        last_channels = make_divisible(last_channels * width_mult) if width_mult > 1.0 else last_channels
        self.features = [conv_3x3_bn(in_channels=3, out_channels=32, stride=2)]
        
        # build inverted residual block
        for t, c, n, s in inverted_residual_setting:
            output_channels = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 1:
                    self.features.append(block(input_channels, output_channels, stride=s, expand_ratio=t))
                else:
                    self.features.append(block(input_channels, output_channels, stride=1, expand_ratio=t))
                
                input_channels = output_channels
                
        # building last several layer
        self.features.append(conv_1x1_bn(input_channels, last_channels))
        # make nn.Sequential
        self.features = nn.Sequential(*self.features)
        
        # building classifier
        self.classifier = nn.Linear(in_features=1028, out_features=num_classes)
        
        self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()        
                
        
def test(device='cpu'):
    net = MobileNetV2(img_size=224, num_classes=100, width_mult=1)
    from torchsummary import summary
    summary(net, input_data=(3, 96, 96), device=device)
    x = torch.randn(64, 3, 96, 96)
    y = net(x).to(device)
    print(y.shape)
    
# test()
        