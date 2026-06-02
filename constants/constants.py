class DataLocations:
    esc_10 = "./ESC-10"
    sound_adapter = "/hdd/subrata/domain_generalization/data/sound_adapter" #"./data/sound_adapter"
    esc_50 = "./ESC-50"


class DomainInfo:
    domain_considered = [2, 3, 6, 7, 9]
    domain_label_map = {}
    for i, domain in enumerate(domain_considered):
        domain_label_map[domain] = i
    folder = "mqtt_rec1"   


class FeatureParams:
    fs = 44100
    hop_len = 256
    win_len = 2 * hop_len
    nfft = 1024
    label_df_loc = "./class_domain_label.csv"
    save_loc = "./extracted_features_esc_50"
    train_csv_loc = "./train.csv"
    val_csv_loc = "./val.csv"
    test_csv_loc = "./test.csv"


class TrainingParams:
    feat_path = "./extracted_features_esc_50"
    save_path = "./saved_results_esc_50"       
    ####save path每次要改
    model_name = "sound_adpater_trained_on_domain_1_cnn_model_deep.h5"
    train_csv_loc = "./train.csv"
    val_csv_loc = "./val.csv"
    test_csv_loc = "./test.csv"
    stat_csv_name = "stat_sa__deep_d_1.csv"
    batch_size = 32
    lr = 1e-3
    max_epoch = 300
    esc_10 = False
