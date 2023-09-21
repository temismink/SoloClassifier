#!/usr/bin/env python
# coding: utf-8
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import os
import subprocess
from pytube import Playlist
from moviepy.editor import VideoFileClip

dataset_directory = "./demucs/solo_dataset"

dataset = AudioDataset(dataset_directory)

batch_size = 32  # You can adjust this based on your needs
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Replace this with the URL of your YouTube playlist
playlist_url = "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"

# Directory to store downloaded videos
output_directory = "./saved_videos"

os.makedirs(output_directory, exist_ok=True)

# Download videos from the playlist
playlist = Playlist(playlist_url)
for video_url in playlist.video_urls:
    video = pytube.YouTube(video_url)
    stream = video.streams.get_highest_resolution()
    video_filename = os.path.join(output_directory, f"{video.title}.mp4")
    stream.download(output_path=output_directory, filename=video_filename)

# Convert downloaded videos to .wav
for filename in os.listdir(output_directory):
    if filename.endswith(".mp4"):
        video_path = os.path.join(output_directory, filename)
        wav_filename = os.path.splitext(filename)[0] + ".wav"
        wav_path = os.path.join(output_directory, wav_filename)
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(wav_path)
        clip.close()
        os.remove(video_path)


import torchaudio
import torchaudio.transforms as transforms

def load_and_preprocess_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    mfcc_transform = transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
    )
    mfcc = mfcc_transform(waveform)
    return mfcc

audio_file = "path_to_audio.wav"
mfcc = load_and_preprocess_audio(audio_file)

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(32 * 53 * 13, 128)  # Adjust input size according to your MFCC features
        self.fc2 = nn.Linear(128, 2)  # Two classes: "solo" and "not solo"

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = []
        self.labels = []

        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for filename in os.listdir(label_dir):
                    file_path = os.path.join(label_dir, filename)
                    self.file_paths.append(file_path)
                    self.labels.append(0 if label == "not_solo" else 1)  # "not solo" is labeled as 0, "solo" is labeled as 1

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,  # Adjust based on your desired number of MFCC coefficients
        )
        mfcc = mfcc_transform(waveform)
        label = self.labels[idx]
        return mfcc, label

model = AudioClassifier()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


model.eval()

true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels in dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.tolist())
        predicted_labels.extend(predicted.tolist())

f1 = f1_score(true_labels, predicted_labels)

print("F1-Score: {:.4f}".format(f1))
