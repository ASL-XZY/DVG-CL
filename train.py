import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from utils import get_dataloaders, load_data
from model import DVG_CL
from CVC_loss import CVCLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train DVG-CL Model with Contrastive and DGI Loss")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--graph_method", type=str, default="GNN", choices=["GNN", "GIN"])
    parser.add_argument("--save_path", type=str, default="contrastive_model.pth")

    parser.add_argument("--dataset_name", type=str, default="FingerMovements",
                        help="Name of dataset subfolder under data/")
    parser.add_argument("--data_dir", type=str, default="/media/h3c/data2/UEA/", help="Parent directory for datasets")

    parser.add_argument("--lambda_node", type=float, default=0.7)
    parser.add_argument("--lambda_graph", type=float, default=0.7)
    parser.add_argument("--lambda_dgi", type=float, default=0.5)
    return parser.parse_args()


def train_model(args):
    print("Starting training with config:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    start_time = datetime.now()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = get_dataloaders(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    (X_train, _), _ = load_data(args.dataset_name, args.data_dir)
    num_nodes, feature_dim = X_train.shape[1], X_train.shape[2]

    model = DVG_CL(num_nodes, feature_dim, use_cuda=torch.cuda.is_available(), graph_method=args.graph_method).to(
        device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    contrastive_loss_fn = CVCLoss(temperature=args.temperature)

    dgi_loss_fn = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_epoch = 0
    wait_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for X, _ in train_loader:
            X = X.to(device)
            optimizer.zero_grad()

            combined_output, gnn_output, hgnn_output, logits, lbl = model(X)
            loss_DGI = dgi_loss_fn(logits, lbl)
            gnn_output = gnn_output.squeeze(1)

            loss_graph, loss_node = contrastive_loss_fn(gnn_output, hgnn_output)

            loss = (
                    args.lambda_node * loss_node +
                    args.lambda_graph * loss_graph +
                    args.lambda_dgi * loss_DGI
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {total_loss:.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            best_epoch = epoch
            wait_counter = 0
            torch.save(model.state_dict(), args.save_path)
            print(f" Model saved at epoch {epoch + 1} with loss {total_loss:.4f}")
        else:
            wait_counter += 1

    print(f"\nBest model saved at epoch {best_epoch + 1} with loss {best_loss:.4f}")
    print(f"Total training time: {datetime.now() - start_time}")


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
