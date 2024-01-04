import os
import matplotlib .pyplot as plt
import numpy as np
import pandas as pd
import librosa
import librosa.display
from dtw import dtw
import scipy


def extract_mfccs(file_path):
    data, sr = librosa.load(file_path)
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def calculate_dtw(file1, file2):
    signal1 = scipy.io.wavfile.read(file1)
    signal2 = scipy.io.wavfile.read(file2)

    x = np.array([2, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
    y = np.array([1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)

    euclidean_norm = lambda x, y: np.abs(x - y)

    dist, cost_matrix, acc_cost_matrix, path = dtw(signal1, signal2, dist=euclidean_norm)

    return dist
    
    

def predict_gender(train_data, train_labels, test_path):
    test_mfccs = extract_mfccs(test_path)
    distances = []
    for i in range(len(train_data)):
        distances.append(calculate_dtw(train_data[i], test_path))
    min_distance = min(distances)
    min_index = distances.index(min_distance)
    return train_labels[min_index]


def evaluate_predictions(predictions, test_data):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == test_data[i][1]:
            correct += 1
    return correct / len(predictions) * 100  #accuracy in percent

def load_data_and_labels():
    data = []
    labels = []
    folder_path = './trainall'
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            mfccs = extract_mfccs(file_path)
            data.append(mfccs)
            label = filename.split('_')[1].strip()
            labels.append(0 if label[0] == 'M' else 1)   
    return data, labels

def main():
    data, labels = load_data_and_labels()
    file_path = input("Enter the path to the file: ")

    while not file_path.endswith(".wav"):
        print("The file must be a .wav file")
        file_path = input("Please enter the correct path to the file: ")

    print("Predicted gender: ", predict_gender(data, labels, file_path))


if __name__ == "__main__":
    main()




