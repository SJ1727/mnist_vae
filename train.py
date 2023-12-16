import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from model import VAE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 3
TRAINING = True
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
transformations = transforms.Compose(
    (
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    )
)
dataset = datasets.MNIST("data", True, transformations, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
net = VAE()
optim = optim.Adam(net.parameters(), LEARNING_RATE)
criterion = nn.MSELoss()
step = 0

for epoch in range(EPOCHS):
    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(DEVICE)
        
        # Pass through VAE and get reconstructions
        reconstructions = net.forward(images)
        
        net.zero_grad()
        
        # Calculate loss and gradients
        loss = criterion(reconstructions, images)
        loss.backward()
        
        # Update Parameters
        optim.step()
        
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}: {loss}")
            
            writer.add_images("Real", images[:16], step)
            writer.add_images("Reconstructred", reconstructions[:16], step)
            writer.add_scalar("Loss", loss, step)

            step += 1

writer.flush()