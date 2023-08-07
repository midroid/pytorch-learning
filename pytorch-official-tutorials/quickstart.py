import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Training Data from open datasets
training_data = datasets.FashionMNIST('./data', train=True, download=True, transform=ToTensor())

test_data = datasets.FashionMNIST('./data', train=False, download=True, transform=ToTensor())

batch_size = 64

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f'shape of X [N, C, H, W]: {X.shape}')
    print(f'shape of y {y.shape} {y.dtype}')
    break

# Creating models
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f'Using {device} device')

# Define Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
# model = NeuralNetwork().to(device)
# print(model)

# Optimizing the parameters
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            print(f"pred: {pred}")
            print(f"y: {y}")
            print(f"loss: {loss}")
            loss, current = loss.item(), (batch+1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            print(f'pred: {pred.argmax(1)}, y: {y}')
            print(pred.argmax(1) == y)
            print(f"correct: {correct}")
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>.8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n---------------")
    # train(train_dataloader, model, loss_fn, optimizer)
    # test(test_dataloader, model, loss_fn)
print("Done")

# Saving Models
# torch.save(model.state_dict(), "model.pth")
# print("Saved pytorch model state to the model.pth")

# Loading Models
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

# Predictions
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[2][0], test_data[2][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    print(pred)
    print(pred[0])
    print(pred[0].argmax(0))
    print(pred.argmax(1))
    print(torch.max(pred, 1))
    _, preds = torch.max(pred, 1)
    print(preds)
    # predicted, actual = classes[pred[0].argmax(0)], classes[y]
    predicted, actual = classes[preds], classes[y]
    print(f"Predicted: {predicted}, Actual: {actual}")





