from torch.utils.data import Dataset
from src.utils import spec_augment
import torch
import numpy as np

class SER_Dataset(Dataset):
    def __init__(self, features, labels, masks, mode="train"):
        self.features = features
        self.labels = labels
        self.mode = mode
        self.masks = masks
        
    def __len__(self):
        return self.features.shape[0]
    
    def smooth_labels(self, labels, factor=0.1):
        labels = labels.astype(np.float32)
        labels *= (1 - factor)
        labels += (factor / labels.shape[0])
        return labels
    
    def __getitem__(self, index):
        if self.mode == "train":
            feature = self.features[index]
            feature = spec_augment(feature)
        else:
            feature = self.features[index]
            
        mask = self.masks[index]
        label = self.labels[index]
        label = self.smooth_labels(label)
        
        sample = {
            "feature":torch.tensor(feature),
            "label":torch.tensor(label),
            "mask":torch.tensor(mask)
        }
        
        return sample