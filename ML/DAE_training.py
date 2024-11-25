import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load training dataset from DAE_training_data.py
dataset = torch.load('training_data.pt')
dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, pin_memory=True)


# Denoising Model
class DenoisingAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1, padding_mode='zeros'),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded_vec = self.decoder(encoded)
        decoded = decoded_vec.squeeze(1)
        return decoded


# Move model and variables to CUDA
model = DenoisingAE()
model = nn.DataParallel(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training
epochs = 10
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
criterion = nn.MSELoss()
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for data in tqdm(dataloader):
        inputs, targets = data
        # inputs = inputs.unsqueeze(1)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step(epoch_loss / len(dataloader))
    print(f'\nEpoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.6f}')
torch.save(model.module.state_dict(), 'denoising_autoencoder.pth')
