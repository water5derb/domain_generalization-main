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
from models.model import CNNModel
from constants.constants import TrainingParams
from utils import create_folder, gpu_to_cpu


def train_epoch(model, data_generator, optimizer, criterion, device, epoch, max_epoch, batch_size):
    model.train()
    train_loss = 0
    class_loss_ = 0
    domain_loss_ = 0
    nb_train_batches = 0
    i = 0
    len_dataloader = len(data_generator)
    for x, class_x, domain_x in tqdm(data_generator, desc=f'Epoch: {epoch+1}/{max_epoch}'):
        p = float(i + epoch * len_dataloader) / max_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        x = torch.tensor(x).to(device).float()
        class_x = torch.tensor(class_x).to(device)
        domain_x = torch.tensor(domain_x).to(device)
        optimizer.zero_grad()
        class_y, domain_y = model(x, alpha)
        class_loss = criterion(class_y, class_x)
        class_loss_ += class_loss
        domain_loss = criterion(domain_y, domain_x)
        domain_loss_ += domain_loss
        loss = class_loss + domain_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        nb_train_batches += 1
        i += 1
    train_loss /= nb_train_batches
    class_loss_ /= nb_train_batches
    domain_loss_ /= nb_train_batches
    return train_loss, class_loss_, domain_loss_


def test_epoch(model, data_generator, criterion, device, desc):
    model.eval()
    test_loss = 0
    class_loss_ = 0
    domain_loss_ = 0
    nb_test_batches = 0
    all_true_class_label, all_pred_class_label = [], []
    all_true_domain_label, all_pred_domain_label = [], []
    with torch.no_grad():
        for x, class_x, domain_x in tqdm(data_generator, desc=desc):
            x = torch.tensor(x).to(device).float()
            class_x = torch.tensor(class_x).to(device)
            domain_x = torch.tensor(domain_x).to(device)
            class_y, domain_y = model(x, alpha=0.)
            class_loss = criterion(class_y, class_x)
            class_loss_ += class_loss
            domain_loss = criterion(domain_y, domain_x)
            domain_loss_ += domain_loss
            loss = class_loss + domain_loss
            test_loss += loss.item()
            nb_test_batches += 1
            class_x, class_y = gpu_to_cpu(class_x), gpu_to_cpu(class_y)
            domain_x, domain_y = gpu_to_cpu(domain_x), gpu_to_cpu(domain_y)
            all_true_class_label += list(class_x)
            all_true_domain_label += list(domain_x)
            all_pred_class_label += list(np.argmax(class_y, axis=1))
            all_pred_domain_label += list(np.argmax(domain_y, axis=1))
    test_loss /= nb_test_batches
    class_loss_ /= nb_test_batches
    domain_loss_ /= nb_test_batches
    accuracy_class = accuracy_score(all_true_class_label, all_pred_class_label)
    accuracy_domain = accuracy_score(all_true_domain_label, all_pred_domain_label)
    return test_loss, class_loss_, domain_loss_, accuracy_class, accuracy_domain


def train():
    training_params = TrainingParams()
    create_folder(training_params.save_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device=device)
    # summary(model, [(1, 513, 1034), (0.)])
    train_loader = load_data(
        batch_size=training_params.batch_size,
        feat_path=training_params.feat_path,
        csv_path=training_params.train_csv_loc,
        shuffle=True
    )
    val_loader = load_data(
        batch_size=training_params.batch_size,
        feat_path=training_params.feat_path,
        csv_path=training_params.val_csv_loc,
        shuffle=False
    )
    test_loader = load_data(
        batch_size=training_params.batch_size,
        feat_path=training_params.feat_path,
        csv_path=training_params.test_csv_loc,
        shuffle=False
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=training_params.lr)
    best_val_epoch, best_val_loss = 0, 1e10
    stats = []
    for epoch in range(training_params.max_epoch):
        train_loss, class_loss, domain_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            training_params.max_epoch, training_params.batch_size
        )
        val_loss, val_class_loss, val_domain_loss, accuracy_class, accuracy_domain = test_epoch(
            model, val_loader, criterion, device, "Validating: "
        )
        print(f'Training Loss: {train_loss} | Validation Loss: {val_loss} | Class Acc: {accuracy_class} | Domain Acc: {accuracy_domain}')
        if val_loss <= best_val_loss:
            torch.save(model.state_dict(), os.path.join(training_params.save_path, training_params.model_name))
            best_val_loss = val_loss
            best_val_epoch = epoch + 1
        stats.append([
            epoch+1, train_loss, class_loss, domain_loss, val_loss, val_class_loss,
            val_domain_loss, accuracy_class, accuracy_domain
        ])
    print(f'Best model saved at {best_val_epoch}th epoch!')
    model.load_state_dict(torch.load(os.path.join(training_params.save_path, training_params.model_name), map_location='cpu'))
    print('Loading unseen test dataset:')
    test_loss, test_class_loss, test_domain_loss, accuracy_class, accuracy_domain = test_epoch(
        model, test_loader, criterion, device, "Testing: "
    )
    stats.append([
        1e3, test_loss, test_class_loss, test_domain_loss, test_loss, test_class_loss,
        test_domain_loss, accuracy_class, accuracy_domain
    ])
    print(f'Test Loss: {test_loss} | Class Acc: {accuracy_class} Domain Acc: {accuracy_domain}')
    stat_df = pd.DataFrame(
        stats,
        columns=[
            "Epoch", "train_loss", "train_class_loss", "train_domain_loss",
            "val_loss", "val_class_loss", "val_domain_loss",
            "class_accuracy", "domain_accuracy"
        ]
    )
    stat_df.to_csv(os.path.join(training_params.save_path, training_params.stat_csv_name), index=False)
