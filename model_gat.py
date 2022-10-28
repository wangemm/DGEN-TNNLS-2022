import torch
import torch.nn as nn
from torch_geometric.nn import GATConv as GAT
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from layers import KPooling
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import time

EPS = 1e-15


class Gae(nn.Module):
    def __init__(self, in_dim, hidden_dims1, hidden_dims2, heads, GP_out_dims, n_clusters):
        super(Gae, self).__init__()
        self.gat_1 = GAT(in_channels=in_dim, out_channels=hidden_dims1, heads=heads[0], dropout=0.5)
        self.gat_2 = GAT(in_channels=hidden_dims1 * heads[0], out_channels=hidden_dims2)

        self.pool = KPooling(in_channels=hidden_dims2, n_clusters=n_clusters)

        self.gat_3 = GAT(in_channels=hidden_dims2, out_channels=hidden_dims2)
        self.gat_4 = GAT(in_channels=hidden_dims2, out_channels=GP_out_dims)

        self.decoder = InnerProductDecoder()
        self.v = 1
        self.GP_cluster_layer = Parameter(torch.Tensor(n_clusters, GP_out_dims))
        torch.nn.init.xavier_normal_(self.GP_cluster_layer.data)

    def encode(self, z, edge_index):
        h = F.elu(self.gat_1(z, edge_index))
        h = self.gat_2(h, edge_index)

        return h

    def GPencode(self, z, edge_index, snn_max):
        h = F.elu(self.gat_1(z, edge_index))
        h = self.gat_2(h, edge_index)

        out, edge_index_d, perm = self.pool(h, edge_index, snn_max)

        out = F.elu(self.gat_3(out, edge_index_d))
        out = self.gat_4(out, edge_index_d)

        # Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(out.unsqueeze(1) - self.GP_cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return out, edge_index_d, perm, q

    def recon_loss(self, z, pos_edge_index):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)

        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss


class Gae_pre(nn.Module):
    def __init__(self, in_dim, hidden_dims, heads, n_clusters):
        super(Gae_pre, self).__init__()
        self.gat_1 = GAT(in_channels=in_dim, out_channels=hidden_dims[0], heads=heads[0], dropout=0.5)
        self.gat_2 = GAT(in_channels=hidden_dims[0] * heads[0], out_channels=hidden_dims[-1])

        self.decoder = InnerProductDecoder()
        self.v = 1
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, hidden_dims[-1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def encode(self, z, edge_index):
        h = F.elu(self.gat_1(z, edge_index))
        h = self.gat_2(h, edge_index)

        # Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(h.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return h, q

    def recon_loss(self, z, pos_edge_index):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)

        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss


class InnerProductDecoder(nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""

    def forward(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class Class_net(nn.Module):
    def __init__(self, in_dim, n_clusters):
        super(Class_net, self).__init__()
        self.gat_1 = GAT(in_channels=in_dim, out_channels=8, heads=8, dropout=0.6)
        self.gat_2 = GAT(in_channels=8 * 8, out_channels=n_clusters, heads=1, concat=False, dropout=0.6)

    def running(self, z, edge_index):
        z = F.dropout(z, p=0.6, training=self.training)
        z = F.elu(self.gat_1(z, edge_index))
        z = F.dropout(z, p=0.6, training=self.training)
        z = self.gat_2(z, edge_index)
        return F.log_softmax(z, dim=1)


class Net(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(Net, self).__init__()
        self.conv1 = GAT(num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GAT(8 * 8, num_classes, heads=1, concat=False,
                         dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
