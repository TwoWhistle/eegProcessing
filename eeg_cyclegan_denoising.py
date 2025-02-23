import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

# Check for M1 compatibility
device = torch.device("cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 16
learning_rate_G = 0.0001
learning_rate_D = 0.0001
epochs = 50
chunk_size = 256  # 1 second EEG chunks at 256 Hz

# Load EEG Data
def load_eeg(file_path):
    data = np.loadtxt(file_path)
    return data

# Dataset for EEG
class EEGDataset(Dataset):
    def __init__(self, data, segment_length=256):
        self.data = data
        self.segment_length = segment_length
        self.segments = self.segment_eeg(data)

    def segment_eeg(self, data):
        segments = []
        for i in range(0, len(data) - self.segment_length, self.segment_length):
            noisy = data[i:i + self.segment_length] + np.random.normal(0, 5, self.segment_length)
            clean = data[i:i + self.segment_length]
            segments.append((noisy, clean))
        return segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        noisy, clean = self.segments[idx]
        return torch.tensor(noisy, dtype=torch.float32), torch.tensor(clean, dtype=torch.float32)

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

    def forward(self, x):
        return self.model(x)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Gradient penalty function
def gradient_penalty(discriminator, real, fake):
    alpha = torch.rand(real.size(0), 1, device=real.device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates, 
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True, retain_graph=True
    )[0]
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


# Initialize models
gen_AB = Generator().to(device)
gen_BA = Generator().to(device)
disc_A = Discriminator().to(device)
disc_B = Discriminator().to(device)

# Loss and Optimizer
adversarial_loss = nn.MSELoss()
cycle_loss = nn.L1Loss()
optimizer_G = optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=learning_rate_G)
optimizer_D = optim.Adam(list(disc_A.parameters()) + list(disc_B.parameters()), lr=learning_rate_D)

# Load dataset
file_path = "justEEG.txt"
data = load_eeg(file_path)
dataset = EEGDataset(data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
print("Starting CycleGAN Training...")
for epoch in range(epochs):
    gen_loss_epoch = 0
    disc_loss_epoch = 0

    for noisy, clean in dataloader:
        noisy, clean = noisy.to(device), clean.to(device)

        # Train Generators (G_A2B and G_B2A)
        optimizer_G.zero_grad()

        fake_clean = gen_AB(noisy)
        recon_noisy = gen_BA(fake_clean)

        fake_noisy = gen_BA(clean)
        recon_clean = gen_AB(fake_noisy)

        # Adversarial loss with label smoothing
        real_labels = torch.full_like(disc_A(clean), 0.9)
        fake_labels = torch.full_like(disc_A(fake_noisy), 0.1)

        loss_gan_A2B = adversarial_loss(disc_B(fake_clean), real_labels)
        loss_gan_B2A = adversarial_loss(disc_A(fake_noisy), real_labels)

        # Cycle consistency loss
        loss_cycle_A = cycle_loss(recon_noisy, noisy)
        loss_cycle_B = cycle_loss(recon_clean, clean)

        # Total generator loss
        gen_loss = loss_gan_A2B + loss_gan_B2A + 10 * (loss_cycle_A + loss_cycle_B)
        gen_loss.backward()
        optimizer_G.step()
        gen_loss_epoch += gen_loss.item()

        # Train Discriminators
        optimizer_D.zero_grad()

        # Real vs. Fake loss with gradient penalty
        real_loss_A = adversarial_loss(disc_A(clean), real_labels)
        fake_loss_A = adversarial_loss(disc_A(fake_noisy.detach()), fake_labels)
        gp_A = 50*gradient_penalty(disc_A, clean, fake_noisy.detach())
        disc_loss_A = (real_loss_A + fake_loss_A) / 2 + 10 * gp_A

        real_loss_B = adversarial_loss(disc_B(noisy), real_labels)
        fake_loss_B = adversarial_loss(disc_B(fake_clean.detach()), fake_labels)
        gp_B = 50*gradient_penalty(disc_B, noisy, fake_clean.detach())
        disc_loss_B = (real_loss_B + fake_loss_B) / 2 + 10 * gp_B

        disc_loss = disc_loss_A + disc_loss_B
        disc_loss.backward()
        optimizer_D.step()
        disc_loss_epoch += disc_loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Generator Loss: {gen_loss_epoch:.4f}, Discriminator Loss: {disc_loss_epoch:.4f}")

# Save models
torch.save(gen_AB.state_dict(), "generator_AB_optimized.pth")
torch.save(gen_BA.state_dict(), "generator_BA_optimized.pth")

print("Training complete. Models saved.")

# Test the model and generate output CSV
print("Generating denoised EEG and calculating band powers...")
def calculate_band_powers(eeg_segment, fs=256):
    from scipy.signal import welch
    freqs, psd = welch(eeg_segment, fs=fs, nperseg=256)
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 100)}
    band_powers = {}
    for band, (low, high) in bands.items():
        band_power = np.trapz(psd[(freqs >= low) & (freqs <= high)], freqs[(freqs >= low) & (freqs <= high)])
        band_powers[band] = band_power
    return band_powers

results = []
with torch.no_grad():
    for noisy, _ in dataset:
        noisy = noisy.to(device).unsqueeze(0)
        denoised = gen_AB(noisy).cpu().numpy().flatten()
        band_powers = calculate_band_powers(denoised)
        results.append(band_powers)

# Convert results to DataFrame and save as CSV
results_df = pd.DataFrame(results)
results_df['Theta/Alpha'] = results_df['Theta'] / results_df['Alpha']
results_df['Beta/Theta'] = results_df['Beta'] / results_df['Theta']
results_df['Gamma/Beta'] = results_df['Gamma'] / results_df['Beta']
results_df.to_csv("EEG_Band_Powers_CycleGAN_Optimized.csv", index=False)

print("Denoising complete. Results saved to 'EEG_Band_Powers_CycleGAN_Optimized.csv'.")
