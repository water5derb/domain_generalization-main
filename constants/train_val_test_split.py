import random
import pandas as pd


class Split:
    def __init__(self):
        #self.filename = "./class_domain_label_subra.csv"
        self.filename = "./data/ESC-50/class_domain_label.csv"
        #self.filename = "./class_domain_label.csv"
        self.df = pd.read_csv(self.filename)
        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self.test_df = pd.DataFrame()
        self.split_dict = {}
        self.base_list = None

    def get_split(self, domain_df):
        train, val, test = [], [], []
        if self.base_list is None:
            for class_label in sorted(domain_df["class_label"].unique()):
                filenames = list(domain_df[domain_df["class_label"] == class_label]["filename"].unique())
                random.shuffle(filenames)
                train += filenames[:28]
                val += filenames[28:32]
                test += filenames[32:]
        else:
            base_train, base_val, base_test = self.base_list
            for class_label in sorted(domain_df["class_label"].unique()):
                filenames = list(domain_df[domain_df["class_label"] == class_label]["filename"].unique())
                train += set(filenames).intersection(base_train)
                val += set(filenames).intersection(base_val)
                test += set(filenames).intersection(base_test)
        return train, val, test

    def get_slice(self, domain, list_):
        df = self.df[self.df["domain_label"] == domain]["filename"].isin(list_).to_frame()
        index = list(df[df["filename"] == True].index)
        return self.df.iloc[index]

    def split(self):
        for i, domain in enumerate(sorted(self.df["domain_label"].unique())):
            domain_df = self.df[self.df["domain_label"] == domain]
            train, val, test = self.get_split(domain_df)
            if i == 0:
                self.base_list = [train, val, test]
            self.split_dict[domain] = {
                "train": train, "val": val, "test": test
            }
        for key, value in self.split_dict.items():
            domain_train_df = self.get_slice(key, value["train"])
            domain_val_df = self.get_slice(key, value["val"])
            domain_test_df = self.get_slice(key, value["test"])
            self.train_df = pd.concat([self.train_df, domain_train_df])
            self.val_df = pd.concat([self.val_df, domain_val_df])
            self.test_df = pd.concat([self.test_df, domain_test_df])
        self.train_df = self.train_df.sample(frac=1).reset_index(drop=True)
        self.val_df = self.val_df.sample(frac=1).reset_index(drop=True)
        self.test_df = self.test_df.sample(frac=1).reset_index(drop=True)
        self.train_df.to_csv("./train.csv", index=False)
        self.val_df.to_csv("./val.csv", index=False)
        self.test_df.to_csv("./test.csv", index=False)
        print(f'Train split: {self.train_df.shape}')
        print(f'Validation split: {self.val_df.shape}')
        print(f'Test split: {self.test_df.shape}')



if __name__ == "__main__":
    splitter = Split()
    splitter.split()