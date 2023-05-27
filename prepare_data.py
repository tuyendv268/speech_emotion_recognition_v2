import yaml
from yaml.loader import SafeLoader
from src.feature_extractor import Feature_Extractor
import audio
import numpy as np
from glob import glob
import os
import pandas as pd
import librosa
from tqdm import tqdm

def prepare_data(config):
    feature_extractor = Feature_Extractor(config=config)
    
    path = config["label_path"]
    label_df = pd.read_csv(
        path, sep="|", dtype={"id":str},
        names=["id", "phoneme","text", "label"])
    print(label_df.head())
    
    path = f'{config["wavs_path"]}/*.wav'
    wavs = glob(path)

    wav_df = pd.DataFrame(wavs, columns=["path"])
    wav_df["id"] = wav_df.path.apply(lambda x: os.path.basename(x).strip(".wav"))
    print(wav_df.head())
    
    df = pd.merge(wav_df, label_df, on="id", how="inner")
    print(df.head())
    
    X, y = [], []

    for index in tqdm(df.index, desc="extract feature"):
        _path = df["path"][index]
        _label = df["label"][index]
        
        _label = config["label"][_label]
        label = np.zeros(len(config["label"].keys()))
        label[_label-1] = 1
        if config["feature"] == "mel":
            _feat = feature_extractor.extract_mel_spectrogram(_path)
            
            _feat = np.expand_dims(_feat.T, axis=0)
        elif config["feature"] == "mfcc":
            _feat = feature_extractor.extract_mfcc(_path)
            
            _feat = np.expand_dims(_feat.T, axis=0)

        X.append(_feat)
        y.append(label)
        
    lengths = [x.shape[1] for x in X]
    max_length = max(lengths)
    if max_length > 256:
        max_length = 256
    masks, features = [], []
    for x in tqdm(X, desc="padding data"):
        if max_length > x.shape[1]:
            features.append(
                np.pad(x, ((0, 0), (0, max_length-x.shape[1]), (0,0)),
                'constant', constant_values=0))
            masks.append([1]*x.shape[1] + [0]*(max_length-x.shape[1]))
        else:
            padding_length = (x.shape[1] - max_length )// 2
            features.append(x[:, padding_length:max_length+padding_length])
            masks.append([1]*max_length)
    
    masks = np.array(masks).astype(np.int8)
    features = np.concatenate(features, axis=0).astype(np.float32)
    labels = np.array(y).astype(np.int8)
    
    features = features.transpose(0, 2, 1)
    print(features.shape)
    
    datas = {
        "features":features,
        "masks":masks,
        "labels":labels
    }

    np.save(config["npy_path"].replace(".npy", f'_{config["feature"]}.npy'), datas, allow_pickle=True)
    print(f'save data to {config["npy_path"]}')
    
if __name__ == "__main__":
    with open("configs/datas/tth_vlsp.yaml", "r") as f:
        config = yaml.load(f, SafeLoader)
        
    prepare_data(config)