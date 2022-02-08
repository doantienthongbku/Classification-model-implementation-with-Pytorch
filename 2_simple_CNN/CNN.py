# Import
from audioop import avg
from math import floor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

    
# Create simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1))
        self.fc = nn.Linear(in_features=32*8*8, out_features=num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device = {}".format(device))

# Hyperparameters
in_channel = 3
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

# Load data
train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Initialize network
model = CNN(in_channels=in_channel, num_classes=num_classes).to(device=device)

# Loss ans optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

# Initialize data to plot
train_acc = []
valid_acc = []
losses = []

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Train data: ", end="")
    else:
        print("Valid data: ", end="")
    
    num_correct = 0
    num_sample = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            scores = model(x)
            _, prediction = scores.max(1)
            num_correct += (prediction == y).sum()
            num_sample += prediction.size(0)
            
        print(f"Got {num_correct} / {num_sample} with accuracy {float(num_correct)/float(num_sample) * 100:.2f} %")
        if loader.dataset.train:
            train_acc.append(round(float(num_correct)/float(num_sample) * 100, 2))
        else:
            valid_acc.append(round(float(num_correct)/float(num_sample) * 100, 2))

    model.train()

# Train network
for epoch in range(num_epochs):
    loss_epoch = []
    
    if epoch == 0:
        checkpoint = {"state_dict": model.state_dict(),
                      "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        # forward
        scores = model(data)
        loss = criterion(scores, target)
        loss_epoch.append(loss.item())
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient descent or Adam step
        optimizer.step()
    
    losses.append(sum(loss_epoch) / len(loss_epoch))
        
    print(f"\nEpoch {epoch + 1}:")
    print(f"Loss at epoch {epoch + 1}: {loss}")
    check_accuracy(train_loader, model)
    check_accuracy(valid_loader, model)
    
# Make plot
plt.plot(losses)
plt.show()
plt.plot(train_acc)
plt.plot(valid_acc)
plt.show()

