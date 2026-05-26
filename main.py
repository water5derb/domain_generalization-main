from constants.get_label import GetLabels
from constants.train_val_test_split import Split
from constants.constants import FeatureParams, DataLocations, TrainingParams
from feature.extraction import ExtractFeature
# from train import train
from train_sound_adapter import train


def main():
    # label = GetLabels()
    # label.get_labels()
    # spliter = Split()
    # spliter.split()
    feat_params = FeatureParams()
    data_params = DataLocations()
    feature = ExtractFeature(
        fs=feat_params.fs,
        nfft=feat_params.nfft,
        hop_len=feat_params.hop_len,
        win_len=feat_params.win_len,
        data_loc=data_params.esc_50,
        save_loc=feat_params.save_loc,
        label_df_loc=feat_params.label_df_loc,
        train_csv_loc=feat_params.train_csv_loc
    )
    feature.generate_features()
    feature.normalize_feature()
    # train()


if __name__ == '__main__':
    main()