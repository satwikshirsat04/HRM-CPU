import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from puzzle_dataset import PuzzleDataset
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed

# Config (replace with actual model params if known)
config = {
    'data_path': './data',
    'batch_size': 4,
    'epochs': 2,
    'lr': 1e-4,
    'input_dim': 128,
    'num_classes': 2
}

# Force CPU
device = torch.device("cpu")

# Dataset
train_dataset = PuzzleDataset(config['data_path'], split='train')
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

# Dummy Model (replace with actual model from utils/functions.py or as needed)
class DummyHRM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = DummyHRM(config['input_dim'], config['num_classes']).to(device)
optimizer = optim.Adam(model.parameters(), lr=config['lr'])
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(config['epochs']):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "hrm_cpu_model.pth")