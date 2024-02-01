import os
import librosa
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import DataLoader, TensorDataset
from oneshot_classifier import AudioClassifier
from sklearn.preprocessing import LabelEncoder

sr = 44100  # Change this to match the actual sampling rate of your audio
feature_length = 100 
# Load the saved model
trained_model_path = "/Users/samuelminkov/Documents/Note_Velocity_Labeller/trained_model.pth"  # Replace with the actual path
#new_files_path = "/Users/samuelminkov/Downloads/oneshot2"  # Replace with the actual path
new_files_path = "/Users/samuelminkov/Desktop/cleaned_non_one_shots"
output_folder_oneshots = "/Users/samuelminkov/Desktop/cleaned_oneshots"
output_folder_nononeshots = "/Users/samuelminkov/Desktop/cleaned_non_one_shots"  

###############################################
# Function to process audio files
def process_files(file_path):
    data = []
    for i, file in enumerate(os.listdir(file_path)):
        if file.endswith(".mp3"):
            audio, _ = librosa.load(os.path.join(file_path, file), sr=sr)
            features = librosa.feature.mfcc(y=audio, sr=sr)
            features = features[:, :feature_length]
            data.append(features.T)  # Transpose to have (time_steps, features), label indicates one-shot or non-one-shot
    return data

processed_test = process_files(new_files_path)

# Split into training and test sets
X_test = processed_test[:300]  # Assuming you want the same number of files as in the training set

# Pad sequences
X_test = pad_sequences(X_test, padding='post', truncating='post', dtype='float32')

# Convert to torch tensor
X_test_tensor_ffnn = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
 # Add channel dimension

# Create output folders if they don't exist
os.makedirs(output_folder_oneshots, exist_ok=True)
os.makedirs(output_folder_nononeshots, exist_ok=True)

input_channels_ffnn = 1  # Assuming grayscale MFCC images
input_time_steps_ffnn = feature_length  # Assuming time steps from your MFCC features
input_features_ffnn = X_test.shape[2]
model_ffnn = AudioClassifier(input_channels_ffnn, input_time_steps_ffnn, input_features_ffnn)

with torch.no_grad():
    model_ffnn.eval()
    outputs_ffnn = model_ffnn(X_test_tensor_ffnn)
    predicted_labels_ffnn = torch.argmax(outputs_ffnn, dim=1)
    #accuracy_ffnn = torch.sum(predicted_labels_ffnn == y_test_tensor_ffnn).item() / len(y_test_tensor_ffnn)

for i, file in enumerate(os.listdir(new_files_path)):
    print(i)
    if file.endswith(".mp3"):
        source_path = os.path.join(new_files_path, file)
        if predicted_labels_ffnn[i] == 1:
            destination_folder = output_folder_oneshots
        else:
            destination_folder = output_folder_nononeshots
        destination_path = os.path.join(destination_folder, file)
        os.replace(source_path, destination_path)

print(f"Sorted into output folders: {output_folder_oneshots} and {output_folder_nononeshots}")