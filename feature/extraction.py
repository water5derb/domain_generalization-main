import os
import itertools
import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
import joblib
import librosa
from tqdm import tqdm
from utils import create_folder, trim_or_pad_audio
from constants.constants import FeatureParams, DataLocations


class ExtractFeature:
    def __init__(self, fs, nfft, hop_len, win_len, data_loc, save_loc, label_df_loc, train_csv_loc, window="hann"):
        self.fs = fs
        self.nfft = nfft
        self.hop_len = hop_len
        self.win_len = win_len
        self.data_loc = data_loc
        self.save_loc = save_loc
        self.window = window
        self.eps = 1e-10
        self.label_df = pd.read_csv(label_df_loc)
        self.train_df = pd.read_csv(train_csv_loc)
        create_folder(self.save_loc)

    def get_spectrogram(self, audio_path):
        audio, fs = librosa.load(audio_path, sr=None)
        audio = (audio - audio.mean()) / (audio.std() + self.eps)
        audio = trim_or_pad_audio(audio)
        stft = librosa.core.stft(
            np.asfortranarray(audio),
            n_fft=self.nfft,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window=self.window
        )
        spectrogram = librosa.amplitude_to_db(np.abs(stft))
        return spectrogram

    def generate_features(self):
        folders = self.label_df["folder"].unique()
        for folder in folders:
            folder_path = os.path.join(self.save_loc, str(folder)) 
            create_folder(folder_path)
            folder_df = self.label_df[self.label_df["folder"] == folder]
            with tqdm(total=folder_df.shape[0], desc=f'Folder: {folder}') as pbar:
                for _, row in folder_df.iterrows():
                    pbar.update(1)
                    #folder_num = str(folder).replace('domain_', '').lstrip('0') or '1'         #audio_path = os.path.join(self.data_loc, str(folder), "mqtt_rec1", row["name"])
                    #audio_path = os.path.join(self.data_loc, folder_num, "mqtt_rec", row["name"])   #audio_path = os.path.join(self.data_loc, str(folder), "mqtt_rec1", row["name"])
                    audio_path = os.path.join(self.data_loc, str(folder), "mqtt_rec1", row["name"])
                    spectrogram = self.get_spectrogram(audio_path=audio_path)
                    feat_path = os.path.join(folder_path, '{}.npy'.format(row["name"].split(".")[0]))
                    np.save(feat_path, spectrogram)

    def normalize_feature(self):
        normalized_features_wts_file = os.path.join(self.save_loc, "spec_scaler")
        spec_scaler = preprocessing.StandardScaler()
        with tqdm(total=self.train_df.shape[0], desc="Fitting Scaler: ") as pbar:
            for _, row in self.train_df.iterrows():
                pbar.update(1)
                feat_path = os.path.join(self.save_loc, str(row["folder"]), '{}.npy'.format(row["name"].split(".")[0]))
                feat_file = np.load(feat_path)
                spec_scaler.partial_fit(feat_file)
                del feat_file
        joblib.dump(
            spec_scaler,
            normalized_features_wts_file
        )
        with tqdm(total=self.label_df.shape[0], desc="Normalizing Features: ") as pbar:
            for _, row in self.label_df.iterrows():
                pbar.update(1)
                feat_path = os.path.join(self.save_loc, str(row["folder"]), '{}.npy'.format(row["name"].split(".")[0]))
                feat_file = np.load(feat_path)
                feat_file = spec_scaler.transform(feat_file)
                np.save(feat_path, np.expand_dims(feat_file, axis=0))
                del feat_file
    
