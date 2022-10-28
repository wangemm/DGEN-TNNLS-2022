from torch_geometric.nn import GraphConv, GATConv
from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_max
import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans as km
from sklearn.mixture import GaussianMixture as GMM
from utils import euclidean_dist
from torch_geometric.utils import to_dense_adj

class SAGPooling(torch.nn.Module):

    def __init__(self, in_channels, ratio=0.8, nonlinearity=torch.tanh):
        super(SAGPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.gat = GATConv(in_channels=in_channels, out_channels=in_channels, heads=4)
        self.linear = nn.Linear(in_channels * 4, 1)
        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        self.gat.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x, edge_index):
        batch = edge_index.new_zeros(x.size(0))

        score = self.linear(F.elu(self.gat(x, edge_index)))
        score = self.nonlinearity(torch.squeeze(score))

        perm = topk(score, self.ratio, batch)
        xr = x[perm] * score[perm].view(-1, 1)
        x = x[perm]

        edge_index = filter_adj(edge_index, perm, num_nodes=score.size(0))

        return xr, edge_index, perm


class KPooling(torch.nn.Module):

    def __init__(self, in_channels, n_clusters, ratio=0.6, nonlinearity=torch.tanh):
        super(KPooling, self).__init__()

        self.n_clusters = n_clusters
        self.in_channels = in_channels
        self.ratio = ratio
        self.gat = GATConv(in_channels=in_channels, out_channels=in_channels, heads=4)
        self.linear = nn.Linear(in_channels * 4, 1)
        self.nonlinearity = nonlinearity
        self.weight = Parameter(torch.Tensor(1,in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.gat.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x, edge_index, snn_max):

        batch = edge_index.new_zeros(x.size(0))
        edge_index_n = edge_index.data.cpu().numpy().T
        score = compute_score(x, self.n_clusters, snn_max)
        perm = topk(score, self.ratio, batch)
        xr = x[perm] * score[perm].view(-1, 1)
        x = x[perm]

        edge_index = filter_adj(edge_index, perm, num_nodes=score.size(0))

        return x, edge_index, perm


def topk(x, ratio, batch):
    num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1),
         num_nodes.cumsum(dim=0)[:-1]], dim=0)

    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes,),
                         torch.finfo(x.dtype).min)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)

    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device) +
        i * max_num_nodes for i in range(batch_size)
    ]
    mask = torch.cat(mask, dim=0)

    perm = perm[mask]

    return perm


def filter_adj(edge_index, perm, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes,), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    return torch.stack([row, col], dim=0)


def compute_score(x, n_clusters, snn_max):

    # kmeans = km(n_clusters=n_clusters, n_init=20)
    # y_pred = kmeans.fit_predict(x.data.cpu().numpy())
    # kc1 = torch.from_numpy(kmeans.cluster_centers_).to(torch.device(x.device))
    # print(kc1)
    gmm = GMM(n_components=n_clusters).fit(x.data.cpu().numpy())
    kc = torch.from_numpy(gmm.means_).to(torch.device(x.device))
    kc = kc.to(torch.float32)
    # print(kc)
    
    x_kc_distance = euclidean_dist(x, kc)
    snn_max = snn_max.long().to(torch.device(x.device))
    zc_index = torch.min(x_kc_distance, 1)[1].type_as(snn_max).to(torch.device(x.device))
    zc_min = torch.min(x_kc_distance, 1)[0].to(torch.device(x.device))
    scores = torch.zeros(x.shape[0]).to(torch.device(x.device))
    for i in range(snn_max.shape[0]):
        scores[i] = torch.nn.functional.pairwise_distance(torch.unsqueeze(x[snn_max[i],:],0), torch.unsqueeze(kc[zc_index[i]],0))
    return (10 - scores - zc_min)


