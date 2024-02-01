import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Step 7: Build and Train Feedforward Neural Network Model
class AudioClassifier(nn.Module):
    def __init__(self, input_channels, input_time_steps, input_features):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(3, 3))
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(32 * ((input_time_steps - 2) // 2) * ((input_features - 2) // 2), 128)
        self.fc2 = nn.Linear(128, 64)  # Additional hidden layer
        self.fc3 = nn.Linear(64, 2)  # Two classes: "solo" and "not solo"

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # Additional hidden layer
        x = self.relu(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # Step 1: Data Preparation
    one_shots_path = "/Users/samuelminkov/Desktop/one_shot_dataset"
    non_one_shots_path = "/Users/samuelminkov/Desktop/non_one_shots"

    one_shots = []
    non_one_shots = []

    # Define the sampling rate and feature length
    sr = 44100  # Change this to match the actual sampling rate of your audio
    feature_length = 100  # Choose a fixed length for your features
    num_files_per_class = 627  # Choose the number of files per class

    def process_files(file_path, label):
        data = []
        for i, file in enumerate(os.listdir(file_path)):
            if file.endswith(".mp3") and i < num_files_per_class:
                audio, _ = librosa.load(os.path.join(file_path, file), sr=sr)
                features = librosa.feature.mfcc(y=audio, sr=sr)
                features = features[:, :feature_length]
                data.append((features.T, label))  # Transpose to have (time_steps, features), label indicates one-shot or non-one-shot
        return data

    one_shots = process_files(one_shots_path, 1)
    non_one_shots = process_files(non_one_shots_path, 0)

    # Step 2: Data Splitting
    data = one_shots + non_one_shots
    X, y = zip(*data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Preprocess Data
    label_encoder = LabelEncoder()
    y_train = to_categorical(label_encoder.fit_transform(y_train))
    y_test = to_categorical(label_encoder.transform(y_test))

    # Pad sequences to the maximum length within each batch
    X_train = pad_sequences(X_train, padding='post', truncating='post', dtype='float32')
    X_test = pad_sequences(X_test, padding='post', truncating='post', dtype='float32')
    # Assuming input_features_ffnn is the number of features from your MFCC features
    input_channels_ffnn = 1  # Assuming grayscale MFCC images
    input_time_steps_ffnn = feature_length  # Assuming time steps from your MFCC features
    input_features_ffnn = X_train.shape[2]  # Use X_train shape to get the number of features

    model_ffnn = AudioClassifier(input_channels_ffnn, input_time_steps_ffnn, input_features_ffnn)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_ffnn.parameters(), lr=0.001)

    # Convert data to torch tensors and create DataLoader
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    y_train_tensor = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Training loop for FFNN
    epochs_ffnn = 1000
    for epoch in range(epochs_ffnn):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model_ffnn(inputs)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate the FFNN model on test set
    X_test_tensor_ffnn = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    y_test_tensor_ffnn = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

    # Save the trained model
    trained_model_path = "/Users/samuelminkov/Documents/Note_Velocity_Labeller/trained_model.pth"  # Replace with the desired path
    torch.save(model_ffnn.state_dict(), trained_model_path)
    print(f"Trained model saved at: {trained_model_path}")

    with torch.no_grad():
        model_ffnn.eval()
        outputs_ffnn = model_ffnn(X_test_tensor_ffnn)
        predicted_labels_ffnn = torch.argmax(outputs_ffnn, dim=1)

        # Calculate metrics
        true_positive = torch.sum((predicted_labels_ffnn == 1) & (y_test_tensor_ffnn == 1)).item()
        false_positive = torch.sum((predicted_labels_ffnn == 1) & (y_test_tensor_ffnn == 0)).item()
        false_negative = torch.sum((predicted_labels_ffnn == 0) & (y_test_tensor_ffnn == 1)).item()
        true_negative = torch.sum((predicted_labels_ffnn == 0) & (y_test_tensor_ffnn == 0)).item()

        # Calculate precision, recall, and F1 score
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Precision (One Shot): {precision * 100:.2f}%")
        print(f"Recall (One Shot): {recall * 100:.2f}%")
        print(f"F1 Score (One Shot): {f1_score * 100:.2f}%")

        # Calculate accuracy for non-one-shot class
        accuracy_non_one_shot = torch.sum(predicted_labels_ffnn == y_test_tensor_ffnn).item() / len(y_test_tensor_ffnn)
        print(f"Accuracy (Non One Shot): {accuracy_non_one_shot * 100:.2f}%")
