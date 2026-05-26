import os
import math
import numpy as np
import pandas as pd


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        print("Folders created")
    else:
        print("Folder already exists!")


def trim_or_pad_audio(audio, t=6, fs=44100):
    max_len = t*fs
    shape = audio.shape
    if shape[0] >= max_len:
        audio = audio[:max_len]
    else:
        n_pad = max_len - shape[0]
        zero_shape = (n_pad,)
        audio = np.concatenate((audio, np.zeros(zero_shape)), axis=0)
    return audio


def gpu_to_cpu(x):
    return x.clone().detach().cpu().numpy()


def slice_dataframe_domain_wise(df, domain):
    df = df[df['domain_label'] == domain].copy()# 加.copy()
    return df

def slice_dataframe_esc10(df, esc10):
    df = df[df["esc10"] == esc10].copy()# 加.copy()
    class_labels = sorted(df["class_label"].unique())
    new_label = {cl: i for i, cl in enumerate(class_labels)}
    df["class_label"] = df["class_label"].map(new_label)  # 直接对列用.map()
    return df
