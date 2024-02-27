import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from normalising_flow import RealNVP

# Define the hyperparameters of the model and the training loop
input_dim = 784
hidden_dim = 256
num_layers = 8
batch_size = 64
learning_rate = 1e-3
num_epochs = 10

# Load the MNIST dataset and create a data loader
train_dataset = MNIST(root="./data", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the flow model and the optimizer
flow_model = RealNVP(input_dim, hidden_dim, num_layers)
optimizer = optim.Adam(flow_model.parameters(), lr=learning_rate)

# Define the loss function to be the negative log-likelihood of the data
def loss_fn(x):
    return -flow_model.log_prob(x).mean()

# Train the flow model
for epoch in range(num_epochs):
    epoch_loss = 0
    for x, _ in tqdm(train_loader):
        # Flatten the input tensor
        x = x.view(x.shape[0], -1)

        # Compute the loss and the gradients
        optimizer.zero_grad()
        loss = loss_fn(x)
        loss.backward()

        # Update the model parameters
        optimizer.step()

        epoch_loss += loss.item()

    print("Epoch {}/{} Loss: {:.4f}".format(epoch + 1, num_epochs, epoch_loss / len(train_loader)))
