import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchaudio.transforms as T

class AudioDataset(Dataset):
    def __init__(self, csv_path, audio_dir, class2idx=None, transform=None):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.transform = transform

        # Tự động tạo class2idx nếu không truyền vào
        if class2idx is None:
            all_labels = sorted(self.df['label'].unique())
            self.class2idx = {label: idx for idx, label in enumerate(all_labels)}
        else:
            self.class2idx = class2idx

        self.num_classes = len(self.class2idx)

        # Tạo mel spectrogram transform
        # THÊM DÒNG NÀY ĐỂ CÓ MEL SPECTROGRAM
        self.mel = T.MelSpectrogram(
            sample_rate=32000,
            n_fft=1024,
            hop_length=320,
            n_mels=224
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['filepath']

        waveform, sr = torchaudio.load(file_path)

        # Convert waveform to mel spectrogram (image-like tensor)
        spec = self.mel(waveform)  # shape: [1, 224, time_steps]

        # Pad/crop to [1, 224, 224]
        if spec.shape[-1] < 224:
            pad = 224 - spec.shape[-1]
            spec = torch.nn.functional.pad(spec, (0, pad))
        spec = spec[:, :, :224]  # Now shape is [1, 224, 224]

        if self.transform:
            spec = self.transform(spec)

        label_idx = self.class2idx[row['label']]
        #label = torch.tensor([label_idx]).long()
        return spec.float(), label_idx

def load_dataset(csv_path, audio_dir, class2idx=None, transform=None, batch_size=16, shuffle=True, num_workers=0):
    dataset = AudioDataset(csv_path, audio_dir, class2idx=class2idx, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def load_single_input(filepath, sr=32000):
    waveform, _ = torchaudio.load(filepath)
    if waveform.shape[-1] < 224:
        pad = 224 - waveform.shape[-1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    waveform = waveform[:, :224]
    waveform = waveform[:, None, :] if waveform.dim() == 2 else waveform
    return waveform.float().unsqueeze(0)
