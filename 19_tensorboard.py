# MNIST
# DataLoader, Transformation
# Multilayer Neural Net, activation function
# Loss and Optimizer
# Training Loop (batch training)
# Model evaluation
# GPU support

# Run tensorboard with `tensorboard --logdir=runs`

import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/mnist2")

# Device config
TORCH_DEVICE_MPS = 'mps'
TORCH_DEVICE_CPU = 'cpu'
TORCH_DEVICE = 'mps'
device = torch.device(TORCH_DEVICE_MPS if torch.backends.mps.is_available() else TORCH_DEVICE_CPU)
print(f'device: {device}')

# hyper parameters
input_size = 784 # 28 * 28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
# learning_rate = 0.001
learning_rate = 0.01

# MNIST
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True
                                        )

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor()
                                        )

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

example = iter(train_loader)
samples, labels = next(example)
print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap='gray')
    
# plt.show()
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_images', img_grid)
writer.close()
# sys.exit()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
    

model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
model.to(device=device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer.add_graph(model, samples.to(device).reshape(-1, 28*28))
writer.close()
# sys.exit()

# training loop
n_total_steps = len(train_loader)

running_loss = 0.0
running_correct_pred = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100, 1, 28, 28
        # 100, 784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predictions = torch.max(outputs, 1)
        running_correct_pred += (predictions == labels).sum().item()

        if ((i + 1) % 100 == 0):
            print(f'epoch {epoch + 1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item(): .4f}')
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('accuracy', running_correct_pred / 100, epoch * n_total_steps + i)
            running_loss = 0.0
            running_correct_pred = 0

# test
labels = []
preds_data = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels1 in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels1 = labels1.to(device)
        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels1.shape[0]
        n_correct += (predictions == labels1).sum().item()

        class_predictions = [F.softmax(outputs, dim=0) for output in outputs]

        labels.append(predictions)
        preds_data.append(class_predictions)


preds = torch.cat([torch.stack(batch) for batch in preds_data])
labels = torch.cat(labels)

acc = 100.0 * n_correct / n_samples
print(f'accuracy: {acc}')

classes = range(10)
for i in classes:
    labels_i = labels == i
    preds_i = preds[:, i]
    print(f'preds_i shape: {preds_i.shape}')
    print(f'labels_i shape: {labels_i.shape}')
    writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
    writer.close()




