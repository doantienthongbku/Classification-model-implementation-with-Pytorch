import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

train_batch_size = 64
train_shuffle = True

valid_batch_size = 64
valid_shuffle = False

# CIFAR-10
# mean = [0.49140018224716187, 0.4821578562259674, 0.44653069972991943]
# std = [0.19525332748889923, 0.19247250258922577, 0.19420039653778076]

# CIFAR-100
mean = [0.5070741176605225, 0.4865521490573883, 0.44091880321502686]
std = [0.20089584589004517, 0.19844239950180054, 0.20229656994342804]

transform = {
    'train': transforms.Compose([
        transforms.Resize(size=(96, 96)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std, inplace=True),
        
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=(96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std, inplace=True)
    ]),
}

train_data = datasets.CIFAR100(
    root='dataset_CIFAR100',
    train=True,
    transform=transform['train'],
    download=True
)

valid_data = datasets.CIFAR100(
    root='dataset_CIFAR100',
    train=False,
    transform=transform['valid'],
    download=True
)

print(f"Have {len(train_data)} train data")
print(f"Have {len(valid_data)} valid data")

train_loader = DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=train_shuffle)
valid_loader = DataLoader(dataset=valid_data, batch_size=valid_batch_size, shuffle=valid_shuffle)
