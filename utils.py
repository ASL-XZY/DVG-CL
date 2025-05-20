import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(dataset_name="HeartBeat", data_dir="/media/h3c/data2/UEA/"):
    path_prefix = os.path.join(data_dir, dataset_name)
    X_train = np.load(os.path.join(path_prefix, "X_train.npy"))
    X_test = np.load(os.path.join(path_prefix, "X_test.npy"))
    y_train = np.load(os.path.join(path_prefix, "y_train.npy"))
    y_test = np.load(os.path.join(path_prefix, "y_test.npy"))

    return (X_train, y_train), (X_test, y_test)


def get_dataloaders(dataset_name="HeartBeat", data_dir="/media/h3c/data2/UEA/", batch_size=32):
    (X_train, y_train), (X_test, y_test) = load_data(dataset_name, data_dir)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def Dot_Graph_Construction(node_features, use_cuda, mask_prob):
    # node features size is (bs, N, dimension)
    # output size is (bs, N, N)
    bs, N, dimen = node_features.size()

    node_features_1 = torch.transpose(node_features, 1, 2)

    Adj = torch.bmm(node_features, node_features_1)

    eyes_like = torch.eye(N).repeat(bs, 1, 1)
    if use_cuda:
        eyes_like = eyes_like.cuda()

    eyes_like_inf = eyes_like * 1e8

    Adj = F.leaky_relu(Adj - eyes_like_inf)

    Adj = F.softmax(Adj, dim=-1)

    # Apply random mask to the adjacency matrix
    mask = (torch.rand(Adj.size()) > mask_prob).float()
    if use_cuda:
        mask = mask.cuda()
    Adj = Adj * mask

    Adj = Adj + eyes_like

    return Adj


def Eu_dis(x):
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def hyperedge_concat(*H_list):
    H = None
    for h in H_list:
        if h is not None and (not isinstance(h, (list, np.ndarray)) or len(h) > 0):  # Check if h is not empty
            # print(f"Processing h with shape: {h.shape if isinstance(h, np.ndarray) else [arr.shape for arr in h]}")
            if H is None:
                H = h
            else:
                if isinstance(h, np.ndarray):
                    if H.shape[0] == h.shape[0]:
                        H = np.hstack((H, h))
                    else:
                        print(f"Incompatible shapes for concatenation: {H.shape} and {h.shape}")
                        return H  # Or handle the shape mismatch appropriately
                elif isinstance(h, list):
                    if isinstance(H, list):
                        tmp = []
                        for a, b in zip(H, h):
                            if a.shape[0] == b.shape[0]:
                                tmp.append(np.hstack((a, b)))
                            else:
                                print(f"Incompatible shapes for concatenation in lists: {a.shape} and {b.shape}")
                                return H  # Or handle the shape mismatch appropriately
                        H = tmp
                    else:
                        print(f"Incompatible types for concatenation: {type(H)} and {type(h)}")
                        return H  # Or handle the type mismatch appropriately
    return H


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False, m_prob=1):
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs=8, split_diff_scale=False, is_probH=False, m_prob=1):
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)

    # hyper_perturbation
    prob = 0.7
    for i in range(H.shape[1]):
        mask = np.random.binomial(1, prob, size=H[:, i].shape)
        H[:, i] = H[:, i] * mask
        H[i, i] = 1

    return H


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)

    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def readout(node_features):
    return torch.sigmoid(torch.mean(node_features, dim=1))
