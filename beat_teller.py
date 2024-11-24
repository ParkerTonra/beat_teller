import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import json
import mirdata
import sys


class AudioDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dataset = mirdata.initialize('giantsteps_tempo')

        # Get all track ids and filter valid ones
        self.track_ids = []
        self.tempos = []

        for track_id in self.dataset.track_ids:
            track = self.dataset.track(track_id)

            # Check if track has valid tempo annotation
            if track.tempo_v2 is not None:
                # Get the first (most confident) tempo
                tempo = track.tempo_v2.tempos[0]
                if tempo > 0:  # Skip files with invalid tempo
                    self.track_ids.append(track_id)
                    self.tempos.append(tempo)

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        tempo = self.tempos[idx]

        try:
            # Load track through mirdata
            track = self.dataset.track(track_id)

            # Load and preprocess audio
            y, sr = track.audio

            # Take first 30 seconds
            if len(y) > sr * 30:
                y = y[:sr * 30]

            # Compute mel spectrogram
            mel_spect = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_mels=128,
                n_fft=2048,
                hop_length=512
            )

            # Convert to log scale
            mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

            # Normalize
            mel_spect = (mel_spect - mel_spect.mean()) / mel_spect.std()

            if self.transform:
                mel_spect = self.transform(mel_spect)

            # Convert to torch tensor
            mel_spect = torch.FloatTensor(mel_spect)
            tempo = torch.FloatTensor([tempo])

            return mel_spect, tempo

        except Exception as e:
            print(f"Error loading track {track_id}: {str(e)}")
            # Return a zero tensor of the correct shape if there's an error
            return torch.zeros((128, 1292)), torch.FloatTensor([0])


class TempoCNN(nn.Module):
    def __init__(self):
        super(TempoCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        try:
            # Add channel dimension
            x = x.unsqueeze(1)

            print(f"Input shape: {x.shape}")
        
            # Convolutional blocks
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.pool(self.relu(self.bn2(self.conv2(x))))
            x = self.pool(self.relu(self.bn3(self.conv3(x))))
            
            # Print shape before flatten
            print(f"Pre-flatten shape: {x.shape}")
            
            # Flatten
            x = x.view(x.size(0), -1)
            
            # Update the size of the first fully connected layer based on actual flatten size
            if not hasattr(self, 'fc1_size_set'):
                self.fc1 = nn.Linear(x.shape[1], 512)
                self.fc1_size_set = True

            # Fully connected layers
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)

            return x
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            raise e


def train_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_tempo_model.pth')


def analyze(file_path):
    try:
        # load the trained model
        model = TempoCNN()
        model.load_state_dict(torch.load('best_tempo_model.pth'))
        model.eval()
        # Process audio file
        y, sr = librosa.load(file_path, duration=30)
        mel_spect = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        mel_spect = (mel_spect - mel_spect.mean()) / mel_spect.std()

        # convert to tensor
        mel_spect = torch.FloatTensor(mel_spect).unsqueeze(0)

        # et prediction
        with torch.no_grad():
            tempo = model(mel_spect)

        print(json.dumps(float(tempo.item())))

    except Exception as e:
        error_result = {
            "error": str(e)
        }
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == "__main__":
    # Create dataset using mirdata
    dataset = AudioDataset()

    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create and train model
    model = TempoCNN()
    train_model(model, train_loader, val_loader)
