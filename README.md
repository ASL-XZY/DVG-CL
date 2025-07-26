# DVG-CL: Dual-view Graph Contrastive Learning for Multivariate Time Series Classification

This repository provides the official implementation of the **DVG-CL** model for multivariate time series classification.


## 🔧 Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
.
├── data                    # The folder for storing data
├── model.py                # DVG-CL model
├── gnn_layer.py            # GNN and HGNN layers
├── utils.py                # Code containing data preprocessing and other operations.
├── CVC_loss.py             # Cross-View Contrasting Loss
├── train.py                # Training script for pretraining DVG-CL
├── downstream.py           # Linear classifier training on learned embeddings
├── requirements.txt
└── README.md
```

## 📊 Dataset

The dataset should be stored in the following structure (supports [UEA Multivariate Time Series Archive](https://timeseriesclassification.com/dataset.php):

```
data/
└── FingerMovements/
    ├── X_train.npy
    ├── y_train.npy
    ├── X_test.npy
    └── y_test.npy
```

Change the `--dataset_name` and `--data_dir` accordingly in your scripts.

## 🚀 Pretraining

```bash
python train.py \
    --dataset_name FingerMovements \
    --data_dir ./data \
    --graph_method GNN \
    --lambda_node 0.7 \
    --lambda_graph 0.7 \
    --lambda_dgi 0.5
```

## 🧠 Downstream Evaluation

```bash
python downstream_no_val.py \
    --dataset_name FingerMovements \
    --data_dir ./data \
    --model_path contrastive_model.pth
```


## 📜 Citation
If you use this code, please cite:
```
@article{XIAO2025107859,
  title={Exploring dual-view graph structures: Contrastive learning with graph and hypergraph for multivariate time series classification},
  author={Ziyi Xiao and Cong Luo and Jiajia Hu and Guodong Sa and Yueyang Wang},
  journal={Neural Networks},
  volume = {192},
  pages = {107859},
  year = {2025}
}
```
