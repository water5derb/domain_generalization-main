import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from data_loader.loader import load_data
from models.model import SounAdapterCNN, CNNDeep, FeatureMLP, HybridCNNFeature
from constants.constants import TrainingParams
from utils import create_folder, gpu_to_cpu

MODEL_MAP = {
    "cnn_deep":         CNNDeep,
    "feature_mlp":      FeatureMLP,
    "hybrid_cnn_feature": HybridCNNFeature,
}


def train_epoch(model, data_generator, optimizer, criterion, device, epoch, max_epoch, batch_size):
    model.train()
    train_loss = 0
    nb_train_batches = 0
    i = 0
    for x, class_x, _ in tqdm(data_generator, desc=f'Epoch: {epoch+1}/{max_epoch}'):
        x = torch.tensor(x).to(device).float()
        class_x = torch.tensor(class_x).to(device)
        optimizer.zero_grad()
        class_y = model(x)
        loss = criterion(class_y, class_x)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        nb_train_batches += 1
        i += 1
    train_loss /= nb_train_batches
    return train_loss


def test_epoch(model, data_generator, criterion, device, desc):
    model.eval()
    test_loss = 0
    nb_test_batches = 0
    all_true_class_label, all_pred_class_label = [], []
    with torch.no_grad():
        for x, class_x, _ in tqdm(data_generator, desc=desc):
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


def train(domain, model_type="cnn_deep"):
    training_params = TrainingParams()
    create_folder(training_params.save_path)
    model_name = f"sound_adapter_{model_type}_domain_{domain}.h5"
    stat_csv_name = f"sound_adapter_{model_type}_stat_domain_{domain}.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = pd.read_csv(training_params.train_csv_loc)["class_label"].nunique()
    model_cls = MODEL_MAP[model_type]
    model = model_cls(class_label=n_classes).to(device=device)
    train_loader = load_data(
        batch_size=training_params.batch_size,
        feat_path=training_params.feat_path,
        csv_path=training_params.train_csv_loc,
        domain=domain,
        esc_10=training_params.esc_10,   #说是这里不用改
        shuffle=True
    )
    val_loader = load_data(
        batch_size=training_params.batch_size,
        feat_path=training_params.feat_path,
        csv_path=training_params.val_csv_loc,
        domain=domain,
        esc_10=training_params.esc_10,    #说是这里不用改
        shuffle=False
    )
    test_loader = load_data(
        batch_size=training_params.batch_size,
        feat_path=training_params.feat_path,
        csv_path=training_params.test_csv_loc,
        domain=domain,
        esc_10=training_params.esc_10,      #说是这里不用改
        shuffle=False
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_params.lr)
    best_val_epoch, best_val_loss = 0, 1e10
    stats = []
    for epoch in range(training_params.max_epoch):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            training_params.max_epoch, training_params.batch_size
        )
        val_loss, accuracy_class = test_epoch(
            model, val_loader, criterion, device, "Validating: "
        )
        print(f'Training Loss: {train_loss} | Validation Loss: {val_loss} | Class Acc: {accuracy_class}')
        if val_loss <= best_val_loss:
            torch.save(model.state_dict(), os.path.join(training_params.save_path, model_name))
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
        stats.append([
            epoch+1, train_loss, val_loss, accuracy_class
        ])
    print(f'Best model saved at {best_val_epoch}th epoch!')
    model.load_state_dict(torch.load(os.path.join(training_params.save_path, model_name), map_location='cpu'))
    print('Loading unseen test dataset:')
    test_loss, accuracy_class = test_epoch(
        model, test_loader, criterion, device, "Testing: "
    )
    stats.append([
        1e3, test_loss, test_loss, accuracy_class
    ])
    print(f'Test Loss: {test_loss} | Class Acc: {accuracy_class} Domain Acc: {accuracy_class}')
    stat_df = pd.DataFrame(
        stats,
        columns=[
            "Epoch", 
            "train_loss", 
            "val_loss", 
            "class_accuracy"
        ]
    )
    stat_df.to_csv(os.path.join(training_params.save_path, stat_csv_name), index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn_deep", choices=list(MODEL_MAP.keys()),
                        help="Model architecture to train")
    parser.add_argument("--domains", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="Domain indices to train on")
    args = parser.parse_args()

    for domain in args.domains:
        train(domain=domain, model_type=args.model)
