import numpy as np
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import adjusted_mutual_info_score as ami_score
from sklearn.metrics.cluster import fowlkes_mallows_score as fmi_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics
import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data


def load_ACM_DBLP(dataset='ACM'):
    if dataset == 'ACM':
        data = np.load('./data/acm.npz')

    elif dataset == 'DBLP':
        data = np.load('./data/dblp.npz')

    x = data['data']
    x = x.astype('float32')
    y = data['labels']
    y = y.astype('int64')
    graph = data['graph'].T
    graph = graph.astype('int64')
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    graph = torch.from_numpy(graph)
    data = Data(x=x,y=y,edge_index=graph)
    return data


def delete_elem(adj, perm, device):
    adj, perm = adj.cpu().numpy(), perm.cpu().numpy()
    arr = np.array([])
    for p in perm:
        a = np.where(adj == p)
        a = np.array(a[1])
        arr = np.append(arr, a)
        arr = arr.astype(np.int32)
    adj = np.delete(adj, arr, axis=1)
    adj, perm = torch.tensor(adj).to(device), torch.tensor(perm).to(device)
    return adj


def delete_y(y, perm):
    perm = perm.cpu().numpy()
    # y = np.delete(y, perm, axis=0)
    # print(y)
    y = y[perm]
    return y


def delete_d(x, perm):
    x, perm = x.cpu().numpy(), perm.cpu().numpy()
    x = np.delete(x, perm, axis=0)
    # print(x)
    return x


def dis_loss_complex(z, edge_index, c, device):
    loss_n = torch.tensor(0).float().to(device)
    loss_p = torch.tensor(0).float().to(device)
    c = torch.tensor(c).float().to(device)
    for i in range(z.shape[0]):
        ind = []
        loss_n += torch.max(torch.mean(c - torch.abs(z[i] - z[np.random.randint(low=0, high=z.shape[0])])),
                            torch.tensor(0).float().to(device)).float()
        for ii in range(edge_index.shape[1]):
            if edge_index[0, ii] == i:
                ind.append(edge_index[1, ii].item())
        for j in ind:
            loss_p += ((z[i] - z[j]) ** 2).mean()
    loss = loss_p + loss_n
    print(loss)
    return loss


def dis_loss(z, adjp, adjn, dp, dn, device):
    lp = dp - adjp
    ln = dn - adjn

    lp = lp.to(device)
    ln = ln.to(device)

    zp = torch.mm(torch.transpose(z, 0, 1), lp)
    zp = torch.mm(zp, z)

    zn = torch.mm(torch.transpose(z, 0, 1), ln)
    zn = torch.mm(zn, z)

    loss = torch.trace(zp) / zp.shape[0] + torch.reciprocal(torch.trace(zn) / zn.shape[0])
    # loss = torch.trace(zp)
    # print(loss)
    return loss


def dis_loss_cos(z, adjp, adjn, dp, dn, device):
    z = torch.mm(z, torch.transpose(z, 0, 1))
    z = torch.reciprocal(z)
    z = torch.mm(adjp.to(device), z).mean()
    return z

def e_loss(z, c, edge_index, device):
    zc_distence = euclidean_dist(z, c)  # 1000*10 , 10*7
    zc_min = torch.min(zc_distence, 1)[0]  # 1000
    zc_mindex = torch.min(zc_distence, 1)[1]  # 1000
    f = torch.FloatTensor(1).to(device)
    for i in range(edge_index.shape[1]):
        assd = torch.nn.functional.pairwise_distance(torch.unsqueeze(z[edge_index[1, i]], dim=0),
                                                   torch.unsqueeze(c[zc_mindex[edge_index[0, i]]], dim=0))
        f+=assd
    return torch.log(f)


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.shape[0], y.shape[0]
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape�?m, 1)，经过expand()方法，扩展n-1次，此时xx的shape�?m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(x, y.t(), beta = 1, alpha = -2)
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩�?    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def compare_1to1(x,y):
    temp = 1
    for i in range(x.shape[0]):
        if x[i] ==1 and y[i] ==1:
            temp+=1
    return temp

#######################################################
# Evaluate Critiron
#######################################################


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc


def best_map(L1, L2):
    # L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def acc_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] == c_x[:])
    accrate = err_x.astype(float) / (gt_s.shape[0])
    return accrate


