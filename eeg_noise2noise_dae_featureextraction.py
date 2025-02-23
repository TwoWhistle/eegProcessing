import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from scipy.signal import welch, butter, lfilter, spectrogram
import pywt
import matplotlib.pyplot as plt

# Define parameters
chunk_size = 256  #256 Hz
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

    band_powers['Total_PSD_Power'] = np.sum(psd)
    for band in bands:
        band_powers[f"{band}_Relative_Power"] = band_powers[band] / band_powers['Total_PSD_Power']

    return band_powers

# Function to calculate spectral entropy
def calculate_spectral_entropy(data, fs=256):
    freqs, psd = welch(data, fs=fs, nperseg=min(256, len(data)))
    psd_norm = psd / np.sum(psd)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    return spectral_entropy

# Function to calculate STFT features
def calculate_spectrogram_features(data, fs=256):
    f, t, Sxx = spectrogram(data, fs)
    stft_features = {
        'STFT_Delta_Mean': np.mean(Sxx[(f >= 0.5) & (f < 4)]),
        'STFT_Theta_Mean': np.mean(Sxx[(f >= 4) & (f < 8)]),
        'STFT_Alpha_Mean': np.mean(Sxx[(f >= 8) & (f < 13)]),
        'STFT_Beta_Mean': np.mean(Sxx[(f >= 13) & (f < 30)]),
        'STFT_Gamma_Mean': np.mean(Sxx[(f >= 30) & (f <= 100)])
    }
    return stft_features

# Function to calculate wavelet features
def calculate_wavelet_features(data):
    coeffs = pywt.wavedec(data, 'db4', level=4)
    wavelet_features = {
        'Wavelet_Delta_Energy': np.sum(np.square(coeffs[4])),
        'Wavelet_Theta_Energy': np.sum(np.square(coeffs[3])),
        'Wavelet_Alpha_Energy': np.sum(np.square(coeffs[2])),
        'Wavelet_Beta_Energy': np.sum(np.square(coeffs[1])),
        'Wavelet_Gamma_Energy': np.sum(np.square(coeffs[0]))
    }
    return wavelet_features

# Dataset class for EEG
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

# Load EEG data
input_file = "justEEG.txt"
eeg_data = np.loadtxt(input_file)

# Prepare dataset and dataloader
dataset = EEGDataset(eeg_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Denoise EEG and extract features
final_denoised = []
with torch.no_grad():
    for x, _ in dataset:
        final_denoised.extend(x.numpy())

# Process denoised EEG in chunks
all_features = []
for i in range(0, len(final_denoised), chunk_size):
    chunk = final_denoised[i:i + chunk_size]
    if len(chunk) < chunk_size:
        break
    filtered_chunk = bandpass_filter(chunk)

    band_powers = calculate_band_powers(filtered_chunk)
    spectral_entropy = calculate_spectral_entropy(filtered_chunk)
    stft_features = calculate_spectrogram_features(filtered_chunk)
    wavelet_features = calculate_wavelet_features(filtered_chunk)

    feature_set = {**band_powers, 'Spectral_Entropy': spectral_entropy, **stft_features, **wavelet_features}
    all_features.append(feature_set)

# Convert results to DataFrame
results_df = pd.DataFrame(all_features)

# Calculate power ratios
if not results_df.empty:
    results_df['Theta/Alpha'] = results_df['Theta'] / results_df['Alpha']
    results_df['Beta/Theta'] = results_df['Beta'] / results_df['Theta']
    results_df['Gamma/Beta'] = results_df['Gamma'] / results_df['Beta']

# Save results to CSV
results_df.to_csv("EEG_Advanced_Features.csv", index=False)

# Display summary
print("EEG feature extraction complete. Results saved to 'EEG_Advanced_Features.csv'.")
print(results_df.describe())

# Plot example
plt.figure(figsize=(14, 10))
plt.subplot(2, 1, 1)
plt.plot(final_denoised[:chunk_size])
plt.title("Final Denoised EEG Signal (Example Chunk)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

freqs, psd = welch(final_denoised[:chunk_size], fs=256, nperseg=256)
plt.subplot(2, 1, 2)
plt.semilogy(freqs, psd)
plt.title("Power Spectral Density (Example Chunk)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")

plt.tight_layout()
plt.show()
