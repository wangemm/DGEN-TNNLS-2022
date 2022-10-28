import os.path as osp
import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from model_gat import Gae, Class_net
from utils import delete_y,  cluster_acc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
manual_seed = 0
os.environ['PYTHONHASHSEED'] = str(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
np.random.seed(manual_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--t', default=10, type=int)
parser.add_argument('--pre_train', type=bool, default=True)
parser.add_argument('--basis_pretrain_path', type=str, default='save_weight/')
# Network Parameters
parser.add_argument('--out_channels', type=int, default=32)
parser.add_argument('--hidden_dims1', type=int, default=256)
parser.add_argument('--hidden_dims2', type=int, default=32)
parser.add_argument('--heads', type=int, nargs='+', default=[8, 8],
                    help='list of feature hidden dimensions')
parser.add_argument('--tol', default=1e-7, type=float)
# Encoder1 Pretrain Parameters
parser.add_argument('--E1_epochs', type=int, default=1000)
parser.add_argument('--E1_lr', type=float, default=0.0001)
# Model Pretrain Parameters
parser.add_argument('--GP_epochs', type=int, default=10)
parser.add_argument('--GP_lr', type=float, default=0.0001)
# Clustering Train Parameters
parser.add_argument('--Cluster_max_epochs', type=int, default=50)
parser.add_argument('--Cluster_lr', type=float, default=0.0001)
parser.add_argument('--lam1', type=float, default=1)
parser.add_argument('--lam2', type=float, default=10)
# Classification Train Parameters
parser.add_argument('--Class_epochs', type=int, default=1000)
parser.add_argument('--Class_lr', type=float, default=0.01)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(args)


def main():

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    dataset = Planetoid(path, 'CiteSeer', transform=T.NormalizeFeatures())
    data = dataset[0]

    x = data.x
    edge_index = data.edge_index
    print(edge_index.shape)
    y = data.y.data.numpy()
    num_features = x.shape[1]
    n_clusters = len(np.unique(y))
    print('n_clusters:', n_clusters)

    snn_max = np.load('./SNNmatrix/CiteSeer_snnmax.npy')
    snn_max = torch.from_numpy(snn_max)
    args.pretrain_path = args.basis_pretrain_path + 'CiteSeer_out_' + str(
        args.out_channels) + '_GPEpochs_10' + '.pkl'
    model_cluster = Gae(in_dim=num_features, hidden_dims1=args.hidden_dims1, hidden_dims2=args.hidden_dims2,
                        heads=args.heads, n_clusters=n_clusters, GP_out_dims=args.out_channels)
    model_class = Class_net(in_dim=num_features, n_clusters=n_clusters)

    if torch.cuda.is_available():
        model_cluster = model_cluster.to(device)
        model_class = model_class.to(device)
        x = x.to(device)
        edge_index = edge_index.to(device)

    if args.pre_train:
        optimizer_E1 = torch.optim.Adam(model_cluster.parameters(), lr=args.E1_lr)

        for epoch in range(args.E1_epochs):
            model_cluster.train()
            optimizer_E1.zero_grad()
            z = model_cluster.encode(x, edge_index)
            loss = model_cluster.recon_loss(z, edge_index)
            loss.backward()
            optimizer_E1.step()
            if (epoch + 1) % args.t == 0:
                z = model_cluster.encode(x, edge_index)
                # obtain init clustering center
                kmeans = KMeans(n_clusters=n_clusters, n_init=20)
                y_pred = kmeans.fit_predict(z.data.cpu().numpy())
                print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch + 1, loss))

        with torch.no_grad():
            z = model_cluster.encode(x, edge_index)
            # obtain init clustering center
            kmeans = KMeans(n_clusters=n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(z.data.cpu().numpy())
            acc = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)
            print('E1 pretrain results: Acc {:.4f}'.format(acc), 'nmi {:.4f}'.format(nmi), 'ari {:.4f}'.format(ari))

        optimizer_GP = torch.optim.Adam(model_cluster.parameters(), lr=args.GP_lr)

        for epoch in range(args.GP_epochs):
            model_cluster.train()
            optimizer_GP.zero_grad()
            z, edge_index_d, _, q = model_cluster.GPencode(x, edge_index, snn_max)
            loss = model_cluster.recon_loss(z, edge_index_d)
            loss.backward()
            optimizer_GP.step()
            if (epoch + 1) % args.t == 0:
                print('Epoch: {:03d}, Loss: {:.4f}'.format(args.E1_epochs + epoch + 1, loss))
                args.pretrain_path = args.basis_pretrain_path + 'CiteSeer_out_' + str(
                    args.out_channels) + '_GPEpochs_' + str(epoch+1) + '.pkl'
                torch.save(model_cluster.state_dict(), args.pretrain_path)
                print("model saved to {}.".format(args.pretrain_path))
    else:
        model_cluster.load_state_dict(torch.load(args.pretrain_path))
        print('load pretrained ae model from', args.pretrain_path)

    optimizer_Cluster = torch.optim.Adam(model_cluster.parameters(), lr=args.Cluster_lr)

    # cluster parameter initiate
    out, _, perm, _ = model_cluster.GPencode(x, edge_index, snn_max)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(out.data.cpu().numpy())
    model_cluster.GP_cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    yp = delete_y(y, perm)
    acc = cluster_acc(yp, y_pred)
    nmi = nmi_score(yp, y_pred)
    ari = ari_score(yp, y_pred)
    print('GP pretrain results', ':Acc {:.4f}'.format(acc), 'nmi {:.4f}'.format(nmi), 'ari {:.4f}'.format(ari))

    y_pred_last = y_pred
    best_acc2 = 0
    best_epoch = 0

    for epoch in range(int(args.Cluster_max_epochs)):

        if epoch % 1 == 0 or epoch == 1:
            _, _, perm, tmp_q = model_cluster.GPencode(x, edge_index, snn_max)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            y_pred = tmp_q.cpu().numpy().argmax(1)
            yp = delete_y(y, perm)
            acc = cluster_acc(yp, y_pred)
            nmi = nmi_score(yp, y_pred)
            ari = ari_score(yp, y_pred)
            if acc > best_acc2:
                best_acc2 = np.copy(acc)
                best_epoch = epoch
                torch.save(model_cluster.state_dict(), args.basis_pretrain_path + 'CiteSeer_clustering.pkl')
            print('best_Iter {}'.format(best_epoch), ':best_Acc2 {:.4f}'.format(best_acc2), 'Iter {}'.format(epoch),
                  ':Acc {:.4f}'.format(acc),
                  ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
            # check stop criterion
            delta_y = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if epoch > 80 and delta_y < args.tol:
                print('Training stopped: epoch=%d, delta_label=%.8f, tol=%.8f' % (epoch, delta_y, args.tol))
                break

        y_pred = torch.tensor(y_pred)
        optimizer_Cluster.zero_grad()

        z, edge_index_d, _, q = model_cluster.GPencode(x, edge_index, snn_max)
        loss_re = model_cluster.recon_loss(z, edge_index_d)
        loss_clu = F.kl_div(q.log(), p, reduction='batchmean')
        loss = loss_re * args.lam1 + loss_clu * args.lam2
        loss.backward()
        optimizer_Cluster.step()

    model_cluster.eval()

    optimizer_class = torch.optim.Adam(model_class.parameters(), lr=args.Class_lr)

    with torch.no_grad():
        model_cluster.load_state_dict(torch.load(args.basis_pretrain_path + 'CiteSeer_clustering.pkl'))
        out, edge_index_d, perm, tmp_q = model_cluster.GPencode(x, edge_index, snn_max)
        y_pred_local = tmp_q.cpu().numpy().argmax(1)
        yp = delete_y(y, perm)
        acc = cluster_acc(yp, y_pred_local)
        nmi = nmi_score(yp, y_pred_local)
        ari = ari_score(yp, y_pred_local)
        print('GP train results', ':Acc {:.4f}'.format(acc), 'nmi {:.4f}'.format(nmi), 'ari {:.4f}'.format(ari))

        # clusters assignments of informative nodes
        xp = x[perm]

    label_total = torch.tensor(y_pred_local, dtype=torch.long).to(device)

    for epoch in range(int(args.Class_epochs)):
        model_class.train()
        optimizer_class.zero_grad()
        z = model_class.running(xp, edge_index_d)
        loss = F.nll_loss(z, label_total)
        loss.backward()
        optimizer_class.step()
        print(' ' + 'Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))

    model_class.eval()
    z = model_class.running(x, edge_index)
    y_pred = z.max(1)[1].data.cpu().numpy()
    acc = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)
    print('Results', ':Acc {:.4f}'.format(acc), 'nmi {:.4f}'.format(nmi), 'ari {:.4f}'.format(ari))


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


if __name__ == '__main__':
    main()
