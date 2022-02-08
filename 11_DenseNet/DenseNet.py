import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math

# Build DenseNet - B (bottleneck)
class Bottleneck(nn.Module):
    
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        
        interChannels = 4 * growthRate
        self.norm1 = nn.BatchNorm2d(num_features=nChannels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               stride=1 , bias=False)
        self.norm2 = nn.BatchNorm2d(num_features=interChannels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               stride=1, padding=1, bias=False)
        
    def forward(self, x):
        y = self.conv1(self.relu1(self.norm1(x)))
        y = self.conv2(self.relu2(self.norm2(y)))
        y = torch.cat((x, y), dim=1)
        return y
    
class SingleLayer(nn.Module):
    
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        
        self.norm1 = nn.BatchNorm2d(num_features=nChannels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=13,
                               stride=1, padding=1, bias=False)
        
    def forward(self, x):
        y = self.conv1(self.relu1(self.norm1(x)))
        y = torch.cat((x, y), dim=1)
        return y


# DenseNet - C (transition)
class Transition(nn.Module):
    
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_features=nChannels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               stride=1 , bias=False)
        
    def forward(self, x):
        y = self.conv1(self.relu1(self.norm1(x)))
        y = F.avg_pool2d(y, 2)
        return y
    

class DenseNet(nn.Module):
    
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()
        
        nDenseBlock = (depth - 4) // 3
        if bottleneck:
            nDenseBlock //= 2
        
        nChannels = 2 * growthRate
        
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlock, bottleneck)
        nChannels += nDenseBlock * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)
        
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlock, bottleneck)
        nChannels += nDenseBlock * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)
        
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlock, bottleneck)
        nChannels += nDenseBlock * growthRate
        
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        
    def forward(self, x):
        y = self.conv1(x)
        y = self.trans1(self.dense1(y))
        y = self.trans2(self.dense2(y))
        y = self.dense3(y)
        y = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(y)), 8))
        y = self.fc(y)
        return y
         
    
    def _make_dense(self, nChannels, growthRate, nDenseBlock, bottleneck):
        layers = []
        for i in range(int(nDenseBlock)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
            
        return nn.Sequential(*layers)
        
    

def test(device='cpu'):
    model = DenseNet(growthRate=12, depth=40, reduction=0.5, nClasses=100, bottleneck=True)
    from torchsummary import summary
    print(model)
    summary(model, input_data=(3, 32, 32), device=device)

    x = torch.randn(64, 3, 32, 32)
    y = model(x).to(device)
    print(y.shape)
    
# test()

device = 'cpu'
model = models.densenet121()
from torchsummary import summary
# print(model)
summary(model, input_data=(3, 224, 224), device=device)
x = torch.randn(64, 3, 32, 32)
y = model(x).to(device)
print(y.shape)


# mean = [0.4914, 0.4822, 0.4465]
# std = [0.2023, 0.1994, 0.2010]

# train_tfms = transforms.Compose([
#     transforms.Resize(size=(112, 112)),
#     transforms.RandomCrop(112, padding=15, padding_mode='reflect'),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=mean, std=std, inplace=True)
# ])
# valid_tfms = transforms.Compose([
#     transforms.Resize(size=(112, 112)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=mean, std=std, inplace=True)
# ])  

