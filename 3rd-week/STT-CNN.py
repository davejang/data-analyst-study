import os
import numpy as np
import librosa, librosa.display
import torch
import torch.nn as nn

train_X = []
test_X = []
train_path = './data/1'
test_path = './data/2'

for filename in os.listdir(train_path):
    data = f'{train_path}/{filename}'
    wav, sr = librosa.load(data, sr=16000)
    mfcc = librosa.feature.mfcc(wav)

for filename in os.listdir(test_path):
    data = f'{test_path}/{filename}'
    wav, sr = librosa.load(data, sr=16000)
    mfcc = librosa.feature.mfcc(wav) 