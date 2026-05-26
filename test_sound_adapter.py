import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from data_loader.loader import load_data
from models.model import SounAdapterCNN, CNNDeep
from constants.constants import TrainingParams
from utils import create_folder, gpu_to_cpu


def test_epoch(model, data_generator, criterion, device, desc):
    model.eval()
    test_loss = 0
    nb_test_batches = 0
    all_true_
    , all_pred_class_label = [], []
    with torch.no_grad():
        for x, class_x, _ in tqdm(data_generator, desc=desc):
            print(f"Domain data shape - x: {x.shape}, class_x: {class_x.shape}")
            print(f"x dtype: {x.dtype}, min: {x.min():.3f}, max: {x.max():.3f}")
            x = torch.tensor(x).to(device).float()
            class_x = torch.tensor(class_x).to(device)
            class_y = model(x)
            loss = criterion(class_y, class_x)
            test_loss += loss.item()
            nb_test_batches += 1
            class_x, class_y = gpu_to_cpu(class_x), gpu_to_cpu(class_y)
            all_true_class_label += list(class_x)
            all_pred_class_label += list(np.argmax(class_y, axis=1))
    test_loss /= nb_test_batches
    accuracy_class = accuracy_score(all_true_class_label, all_pred_class_label)
    return test_loss, accuracy_class


def test(domain_trained_on):
    print(f"Model Trained on domain {domain_trained_on}")
    model_name = f"sound_adapter_cnn_deep_domain_{domain_trained_on}.h5"
    training_params = TrainingParams()
    create_folder(training_params.save_path)
    domains = [0, 1, 2, 3, 4]
    accuracies = [domain_trained_on]
    for domain in domains:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNNDeep(class_label=10).to(device=device)
        criterion = nn.CrossEntropyLoss()
        test_loader = load_data(
            batch_size=training_params.batch_size,
            feat_path=training_params.feat_path,
            csv_path=training_params.test_csv_loc,
            domain=domain,
            esc_10=training_params.esc_10,
            shuffle=False
        )
        model.load_state_dict(torch.load(os.path.join(training_params.save_path, model_name), map_location='cpu'))
        test_loss, accuracy_class = test_epoch(
            model, test_loader, criterion, device, "Testing: "
        )
        accuracies.append(accuracy_class)
        print(f'Domain: {domain} | Test Loss: {test_loss} | Class Acc: {accuracy_class}')
    return accuracies


if __name__ == "__main__":
    domain_wise_accuracy = []
    domains = [0, 1, 2, 3, 4]
    for domain in domains:
        accuracies = test(domain_trained_on=domain)
        domain_wise_accuracy.append(accuracies)
    result_df = pd.DataFrame(
        domain_wise_accuracy,
        columns=["trained_on", "0", "1", "2", "3", "4"]
    )
    training_params = TrainingParams()
    print(os.path.join(training_params.save_path, "sound_adapter_domain_wise_performance_with_deep_model.csv"))
    result_df.to_csv(os.path.join(training_params.save_path, "sound_adapter_domain_wise_performance_with_deep_model.csv"), index=False)