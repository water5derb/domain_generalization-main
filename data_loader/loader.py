import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils import slice_dataframe_domain_wise, slice_dataframe_esc10


class CustomLoader(Dataset):
    def __init__(self, feat_path, csv_path, domain=None, esc_10=False):
        self.feat_path = feat_path
        self.csv_path = csv_path
        df = pd.read_csv(self.csv_path)
        if domain is not None:
            df = slice_dataframe_domain_wise(df, domain)
        if esc_10 == True:
            df = slice_dataframe_esc10(df, esc_10)
        self.info_list = df.values.tolist()
        del df

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, item):
        _, class_label, domain_label, folder, name, _ = self.info_list[item]  # uncoment for ESC-50
        #_, class_label, domain_label, folder, name = self.info_list[item]##
        _, class_label, domain_label, folder, name, esc10 = self.info_list[item] # uncomment for sound_adapter
        feat_loc = os.path.join(self.feat_path, str(folder), '{}.npy'.format(name.split(".")[0]))
        feat = np.load(feat_loc)
        return feat, class_label, domain_label


def load_data(batch_size, feat_path, csv_path, domain, esc_10, shuffle):
    dataset = CustomLoader(
        feat_path=feat_path,
        csv_path=csv_path,
        domain=domain,
        esc_10=esc_10
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return data_loader
