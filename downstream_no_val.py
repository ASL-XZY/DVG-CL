import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_dataloaders, load_data
from model import DVG_CL
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DVG-CL via downstream classifier")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for feature extraction")
    parser.add_argument("--graph_method", type=str, default="GNN", choices=["GNN", "GIN"], help="Graph encoder type")
    parser.add_argument("--model_path", type=str, default="contrastive_model.pth",
                        help="Path to pretrained contrastive model")
    parser.add_argument("--dataset_name", type=str, default="FingerMovements", help="Name of the dataset folder")
    parser.add_argument("--data_dir", type=str, default="/media/h3c/data2/UEA/", help="Directory where datasets are stored")
    return parser.parse_args()


class DownstreamModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DownstreamModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input features
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def extract_features(model, data_loader, device):
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            combined_output, gnn_output, hgnn_output, _, _ = model(X)
            features_list.append(combined_output.cpu())
            labels_list.append(y.cpu())
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return features, labels.long()


def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total


def compute_metrics(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    precision = precision_score(labels.cpu(), predicted.cpu(), average='macro')
    recall = recall_score(labels.cpu(), predicted.cpu(), average='macro')
    f1 = f1_score(labels.cpu(), predicted.cpu(), average='macro')
    return precision, recall, f1


def evaluate_model(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            if y.dim() != 1:
                y = y.view(-1).long()  # Ensure y is of type long
            outputs = model(X).float()
            loss = criterion(outputs, y)
            total_loss += loss.item() * y.size(0)
            all_outputs.append(outputs)
            all_labels.append(y)
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / all_labels.size(0)
    acc = accuracy(all_outputs, all_labels)
    precision, recall, f1 = compute_metrics(all_outputs, all_labels)
    return acc, avg_loss, precision, recall, f1


def train_downstream_model(features, labels, num_classes, test_features, test_labels, device):
    print(f"Unique labels: {torch.unique(labels)}")
    print(f"Number of classes: {num_classes}")
    # Ensure labels are 1D tensor
    if labels.dim() != 1:
        labels = labels.view(-1).long()  # Ensure labels are of type long

    # Debug: Print shapes of features and labels
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    X_train, y_train = features, labels
    X_test, y_test = test_features, test_labels

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    input_dim = features.shape[1] * features.shape[2]
    model = DownstreamModel(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 200
    best_test_accuracy = 0.0
    best_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        all_train_outputs = []
        all_train_labels = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).long()  # Ensure y_batch is of type long
            # Ensure y_batch is 1D tensor
            if y_batch.dim() != 1:
                y_batch = y_batch.view(-1)
            optimizer.zero_grad()
            outputs = model(X_batch).float()
            loss = criterion(outputs, y_batch.long())
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * y_batch.size(0)
            all_train_outputs.append(outputs)
            all_train_labels.append(y_batch)

        all_train_outputs = torch.cat(all_train_outputs)
        all_train_labels = torch.cat(all_train_labels)
        train_loss = total_train_loss / all_train_labels.size(0)
        train_accuracy = accuracy(all_train_outputs, all_train_labels)

        test_accuracy, test_loss, precision, recall, f1 = evaluate_model(model, test_loader, device, criterion)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_epoch = epoch + 1

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, '
            f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%, '
            f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')

    print(f'Test Accuracy: {best_test_accuracy * 100:.2f}% at Epoch: {best_epoch}')
    return model


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (X_train, y_train), (X_test, y_test) = load_data(args.dataset_name, args.data_dir)
    num_nodes = X_train.shape[1]
    feature_dim = X_train.shape[2]
    num_classes = len(np.unique(y_train))  # Get the number of unique classes

    # Load trained contrastive model
    contrastive_model = DVG_CL(num_nodes=num_nodes, feature_dim=feature_dim,
                               use_cuda=torch.cuda.is_available(), graph_method=args.graph_method).to(device)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    contrastive_model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Extract features from training data
    train_loader, test_loader = get_dataloaders(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    features, labels = extract_features(contrastive_model, train_loader, device)
    test_features, test_labels = extract_features(contrastive_model, test_loader, device)
    # Debug: Print shapes of features and labels
    print(f"Extracted Features shape: {features.shape}")
    print(f"Extracted Labels shape: {labels.shape}")

    # Ensure the number of features and labels match
    assert features.shape[0] == labels.shape[0], "Mismatch between number of features and labels"

    # Train downstream model
    num_classes = len(torch.unique(labels))
    trained_downstream_model = train_downstream_model(features, labels, num_classes, test_features, test_labels,
                                                      device)
