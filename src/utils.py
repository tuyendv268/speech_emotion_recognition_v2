import random
import numpy as np
import librosa
import pandas as pd
import os

def load_data(path):
    data = np.load(path, allow_pickle=True).item()
    
    mels = data["mels"]
    masks = data["masks"]
    labels = data["labels"]
    
    return mels, masks, labels
    
def spec_augment(spec: np.ndarray, num_mask=2, 
                 freq_masking_max_percentage=0.1, time_masking_max_percentage=0.1):
    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)
        
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0
    
    return spec


def load_npy_data(path):
    data = np.load(path, allow_pickle=True).item()
    return data["x"].transpose(0, 2, 1), data["y"]