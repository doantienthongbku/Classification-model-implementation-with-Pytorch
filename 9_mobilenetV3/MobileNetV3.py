import torch
import torch.nn as nn
import torchvision.models as models

device = 'cpu'
num_classes = 100

model = models.mobilenet_v3_large(pretrained=True, width_mult=1.0, num_classes=1000)

model.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes),
        )


def test():
    from torchsummary import summary
    print(model)
    summary(model, input_data=(3, 96, 96), device=device)

    x = torch.randn(64, 3, 96, 96)
    y = model(x).to(device)
    print(y.shape)
    
test()
