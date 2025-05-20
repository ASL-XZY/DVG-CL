from torch.nn import Parameter
from utils import *
from gnn_layer import *


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)


class CNNFeatureExtractor(nn.Module):

    def __init__(self, num_nodes, feature_dim):
        super(CNNFeatureExtractor, self).__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        kernel_ = [5, 7, 9]
        channel = 1
        self.c1 = nn.Conv2d(1, channel, kernel_size=(1, kernel_[0]), stride=1)
        self.c2 = nn.Conv2d(1, channel, kernel_size=(1, kernel_[1]), stride=1)
        self.c3 = nn.Conv2d(1, channel, kernel_size=(1, kernel_[2]), stride=1)

        self.d = (len(kernel_) * self.feature_dim - sum(kernel_) + len(kernel_)) * channel

        self.batch_norm_cnn = nn.BatchNorm1d(self.num_nodes)

    def cnn_act(self, x):
        return F.relu(x)

    def forward(self, x):
        x = x.float()
        a1 = self.c1(x.unsqueeze(1)).reshape(x.shape[0], x.shape[1], -1)
        a2 = self.c2(x.unsqueeze(1)).reshape(x.shape[0], x.shape[1], -1)
        a3 = self.c3(x.unsqueeze(1)).reshape(x.shape[0], x.shape[1], -1)

        x = self.cnn_act(torch.cat([a1, a2, a3], 2))
        x = self.batch_norm_cnn(x)
        return x


class GIN(nn.Module):
    def __init__(self, ginnn, dropout=0.2):
        super(GIN, self).__init__()
        self.dropout = dropout
        self.conv1 = DeGINConv(ginnn)
        ginnn2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        self.conv2 = DeGINConv(ginnn2)

    def forward(self, x, adj):
        h = F.relu(self.conv1(x, adj))
        full_graph_feature = readout(h)
        return full_graph_feature, h


class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super(GraphNeuralNetwork, self).__init__()
        self.dropout = dropout
        self.conv1 = DenseGraphConv(input_dim, 128)
        self.conv2 = DenseGraphConv(128, 128)
        self.conv3 = DenseGraphConv(64, 32)

    def forward(self, x, adj):
        h = F.relu(self.conv1(x, adj))
        full_graph_feature = readout(h)
        return full_graph_feature, h


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class HypergraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super(HypergraphNeuralNetwork, self).__init__()
        self.dropout = dropout
        self.hgnn1 = HGNN_conv(input_dim, 128)
        self.hgnn2 = HGNN_conv(128, 64)
        self.hgnn3 = HGNN_conv(64, 32)

    def forward(self, x, G):
        h = F.relu(self.hgnn1(x, G))
        full_graph_feature = readout(h)
        return full_graph_feature, h


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class DGI(nn.Module):
    def __init__(self, n_in, n_h):
        super(DGI, self).__init__()
        self.gcn = DenseGraphConv(n_in, n_h)
        ginnn = nn.Sequential(
            nn.Linear(n_in, n_h),
            nn.Tanh(),
        )
        self.gin = DeGINConv(ginnn)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj)
        # h_1 = self.gin(seq1, adj)
        c = self.read(h_1, msk)
        c = self.sigm(c)

        h_2 = self.gcn(seq2, adj)
        # h_2 = self.gin(seq2, adj)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    def embed(self, seq, adj, sparse, msk):
        h_1 = F.relu(self.gcn(seq, adj, sparse))
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()


class DVG_CL(nn.Module):
    def __init__(self, num_nodes, feature_dim, use_cuda, graph_method):
        super(DVG_CL, self).__init__()
        self.use_cuda = use_cuda
        self.graph_method = graph_method
        self.cnn = CNNFeatureExtractor(num_nodes, feature_dim)
        self.gnn = GraphNeuralNetwork(self.cnn.d)
        ginnn = nn.Sequential(
            nn.Linear(self.cnn.d, 128),
            nn.Tanh(),
        )
        self.gin = GIN(ginnn)

        self.hgnn = HypergraphNeuralNetwork(self.cnn.d)
        self.batch_norm_gnn = nn.BatchNorm1d(num_nodes)

        self.DGI = DGI(self.cnn.d, 128)

        self.b = Parameter(torch.Tensor(128, 1))
        nn.init.xavier_uniform_(self.b.data)

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        features = self.cnn(x)
        adj = Dot_Graph_Construction(features, self.use_cuda, mask_prob=0.1).to(device)

        batch_size, nb_nodes = x.size(0), x.size(1)
        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]

        lbl_1 = torch.ones(batch_size, nb_nodes)
        lbl_2 = torch.zeros(batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        shuf_fts = shuf_fts.to(device)
        lbl = lbl.to(device)
        logits = self.DGI(features.to(device), shuf_fts, adj, True, None, None, None)

        if self.graph_method == 'GNN':
            _, gnn_output = self.gnn(features.to(device), adj.to(device))

        elif self.graph_method == 'GIN':

            _, gnn_output = self.gin(features.to(device), adj.to(device))

        gnn_output = self.batch_norm_gnn(gnn_output)

        hgnn_outputs = []
        hypergraph = []

        for i in range(features.shape[0]):
            H = construct_H_with_KNN(features[i].cpu().detach().numpy())

            G = generate_G_from_H(H)
            # Ensure G is a tensor
            G = torch.tensor(G, dtype=torch.float32).to(device)
            _, hgnn_output = self.hgnn(features[i].to(device), G)
            hgnn_outputs.append(hgnn_output)
            hypergraph.append(G)

        hgnn_output = torch.stack(hgnn_outputs)
        hgnn_output = self.batch_norm_gnn(hgnn_output)

        # attention
        H = torch.stack([gnn_output, hgnn_output], dim=2)
        b = self.b.unsqueeze(0).unsqueeze(0).repeat(H.size(0), H.size(1), 1, 1)
        beta = torch.softmax(torch.matmul(H, b).squeeze(-1), dim=2)
        combined_output = torch.sum(beta.unsqueeze(-1) * H, dim=2)

        return combined_output, gnn_output, hgnn_output, logits, lbl
