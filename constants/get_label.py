import os
import pandas as pd
from constants import constants


class GetLabels:
    def __init__(self):
        self.data_location = constants.DataLocations
        self.domain_info = constants.DomainInfo
        self.class_label_df = pd.DataFrame(columns=["filename", "class_label"])
        self.class_domain_label_df = pd.DataFrame(
            columns=["filename", "class_label", "domain_label", "folder", "name"]
        )

    def get_class_labels(self):
        folders = [
            folder for folder in os.listdir(self.data_location.esc_10)
            if os.path.isdir(os.path.join(self.data_location.esc_10, folder))
        ]
        folders = sorted(folders, key=lambda x: (isinstance(x, str), x))[:10]
        all_filenames = []
        all_class_labels = []
        for folder in folders:
            file_names = os.listdir(os.path.join(self.data_location.esc_10, folder))
            file_names = [f'{filename.split(".")[0]}.wav' for filename in file_names]
            class_labels = [int(folder.split("-")[0]) - 1]*len(file_names)
            all_filenames += file_names
            all_class_labels += class_labels
        self.class_label_df["filename"] = all_filenames
        self.class_label_df["class_label"] = all_class_labels
        self.class_label_df = self.class_label_df.sample(frac=1)
        # self.class_label_df.to_csv("./constants/class_label.csv", index=False)
    
    def get_class_labels_esc50(self):
        metadata_df = pd.read_csv(os.path.join(self.data_location.esc_50, "esc50.csv"))
        filenames = []
        targets = []
        filenames.extend(metadata_df["filename"].tolist())
        targets.extend(metadata_df["target"].tolist())
        self.class_label_df["filename"] = filenames
        self.class_label_df["class_label"] = targets
    
    def get_domain_label_esc50(self):
        class_df = self.class_label_df
        domains = [folder for folder in os.listdir(self.data_location.esc_50) if os.path.isdir(os.path.join(self.data_location.esc_50, folder))]
        self.domain_label_map = {}
        for i, domain in enumerate(domains):
            self.domain_label_map[domain] = i
        for folder in domains:
            temp_df = pd.DataFrame(columns=["filename", "domain_label", "folder"])
            file_names = os.listdir(os.path.join(
                self.data_location.esc_50, folder)
            )
            names = []
            for filename in file_names:
                names.append(filename)
            domain_labels = [self.domain_label_map[folder]]*len(file_names)
            temp_df["filename"] = names
            temp_df["folder"] = [folder]*len(file_names)
            temp_df["domain_label"] = domain_labels
            temp_df["name"] = file_names
            temp_df1 = pd.merge(class_df, temp_df, on="filename", how="right")
            self.class_domain_label_df = pd.concat([self.class_domain_label_df, temp_df1], ignore_index=True)
        self.class_domain_label_df = self.class_domain_label_df.sample(frac=1)
        self.class_domain_label_df.to_csv(os.path.join(self.data_location.esc_50, 'class_domain_label.csv'), index=False)


    def get_domain_label(self):
        class_df = self.class_label_df
        for folder in self.domain_info.domain_considered:
            temp_df = pd.DataFrame(columns=["filename", "domain_label", "folder"])
            file_names = os.listdir(os.path.join(
                self.data_location.sound_adapter, str(folder), self.domain_info.folder)
            )
            names = []
            for filename in file_names:
                names.append(f'{"-".join(filename.split(".")[0].split("-")[:-1])}.wav')
            domain_labels = [self.domain_info.domain_label_map[folder]]*len(file_names)
            temp_df["filename"] = names
            temp_df["folder"] = [folder]*len(file_names)
            temp_df["domain_label"] = domain_labels
            temp_df["name"] = file_names
            temp_df1 = pd.merge(class_df, temp_df, on="filename", how="right")
            self.class_domain_label_df = pd.concat([self.class_domain_label_df, temp_df1], ignore_index=True)
        self.class_domain_label_df = self.class_domain_label_df.sample(frac=1)
        self.class_domain_label_df.to_csv('./constants/class_domain_label.csv', index=False)

    def get_labels(self):
        self.get_class_labels_esc50()
        self.get_domain_label_esc50()

