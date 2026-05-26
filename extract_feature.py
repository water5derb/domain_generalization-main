from constants.constants import FeatureParams, DataLocations
from feature.extraction import ExtractFeature


def extract():
    feat_params = FeatureParams()
    data_params = DataLocations()
    feature = ExtractFeature(
        fs=feat_params.fs,
        nfft=feat_params.nfft,
        hop_len=feat_params.hop_len,
        win_len=feat_params.win_len,
        data_loc=data_params.sound_adapter,
        save_loc=feat_params.save_loc,
        label_df_loc=feat_params.label_df_loc,
        train_csv_loc=feat_params.train_csv_loc
    )
    feature.generate_features()
    feature.normalize_feature()


if __name__ == "__main__":
    extract()