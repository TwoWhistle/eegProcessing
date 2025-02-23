import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.signal import welch, butter, lfilter
import matplotlib.pyplot as plt

# Define parameters
chunk_size = 256 * 2  # 2 seconds per chunk at 256 Hz
fs = 256  # Sampling frequency

# Bandpass filter function
def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=256, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# Function to calculate band powers
def calculate_band_powers(data, fs=256):
    freqs, psd = welch(data, fs=fs, nperseg=min(256, len(data)))
    bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 100)}
    band_powers = {}

    for band, (low, high) in bands.items():
        band_power = np.trapz(psd[(freqs >= low) & (freqs <= high)], freqs[(freqs >= low) & (freqs <= high)])
        band_powers[band] = band_power

    return band_powers

# Dataset class for Noise2Noise
class EEGDataset(Dataset):
    def __init__(self, data, segment_length=256):
        self.data = data
        self.segment_length = segment_length
        self.segments = self.segment_eeg(data)

    def segment_eeg(self, data):
        segments = []
        for i in range(0, len(data) - self.segment_length, self.segment_length):
            segment1 = data[i:i + self.segment_length] + np.random.normal(0, 5, self.segment_length)
            segment2 = data[i:i + self.segment_length] + np.random.normal(0, 5, self.segment_length)
            segments.append((segment1, segment2))
        return segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        x, y = self.segments[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Noise2Noise model
class Noise2NoiseEEG(nn.Module):
    def __init__(self):
        super(Noise2NoiseEEG, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# Load EEG data
input_file = "justEEG.txt"  # Ensure this file is in the same folder as the script
eeg_data = np.loadtxt(input_file)

# Prepare dataset and dataloader
dataset = EEGDataset(eeg_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train Noise2Noise model
model = Noise2NoiseEEG()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Training Noise2Noise model...")
for epoch in range(5):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/5, Loss: {total_loss / len(dataloader):.4f}")

# Denoise EEG data
model.eval()
denoised_eeg = []
with torch.no_grad():
    for x, _ in dataset:
        x = torch.tensor(x).unsqueeze(0).unsqueeze(0)
        output = model(x)
        denoised_eeg.extend(output.squeeze().numpy())

# Process denoised EEG in chunks
all_band_powers = []
for i in range(0, len(denoised_eeg), chunk_size):
    chunk = denoised_eeg[i:i + chunk_size]
    if len(chunk) < chunk_size:
        break
    filtered_chunk = bandpass_filter(chunk)
    band_powers = calculate_band_powers(filtered_chunk)
    all_band_powers.append(band_powers)

# Convert results to DataFrame
results_df = pd.DataFrame(all_band_powers)

# Calculate power ratios
if not results_df.empty:
    results_df['Theta/Alpha'] = results_df['Theta'] / results_df['Alpha']
    results_df['Beta/Theta'] = results_df['Beta'] / results_df['Theta']
    results_df['Gamma/Beta'] = results_df['Gamma'] / results_df['Beta']

# Save results to CSV
results_df.to_csv("EEG_Band_Powers.csv", index=False)

# Display summary
print("EEG analysis complete. Results saved to 'EEG_Band_Powers.csv'.")
print(results_df.describe())

# Plot example
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
plt.plot(denoised_eeg[:chunk_size])
plt.title("Denoised EEG Signal (Example Chunk)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

freqs, psd = welch(denoised_eeg[:chunk_size], fs=256, nperseg=256)
plt.subplot(2, 1, 2)
plt.semilogy(freqs, psd)
plt.title("Power Spectral Density (Example Chunk)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")

plt.tight_layout()
plt.show()
