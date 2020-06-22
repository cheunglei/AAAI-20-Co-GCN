import numpy as np
import scipy.sparse as sp
import torch
import json
import urllib.request

from torch.utils.data import Dataset

#
# def to_sparse(x):
#     """ converts dense tensor x to sparse format """
#     x_typename = torch.typename(x).split('.')[-1]
#     sparse_tensortype = getattr(torch.sparse, x_typename)
#
#     indices = torch.nonzero(x)
#     if len(indices.shape) == 0:  # if all elements are zeros
#         return sparse_tensortype(*x.shape)
#     indices = indices.t()
#     values = x[tuple(indices[i] for i in range(indices.shape[0]))]
#     return sparse_tensortype(indices, values, x.size())


class MyDataset(Dataset):
    def __init__(self, file_path, dataset_split, view_size=2,):
        super(MyDataset, self).__init__()
        self.file_path = file_path
        self.view_size = view_size
        self.dataset_split = dataset_split

    def __len__(self):
        return len(self.dataset_split)

    def __getitem__(self, item):
        file_idx = self.dataset_split[item]

        adjs, features, y = self.load_data(str(file_idx))

        return adjs, features, y

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def load_data(self, filename,  directed=False):
        """Load citation network dataset (cora only for now)"""
        # print('Loading {} dataset...'.format(dataset))

        idx_features = np.genfromtxt(f"{self.file_path}/{filename}.var",dtype=np.dtype(str))
        # print(idx_features)
        y = np.genfromtxt(f"{self.file_path}/{filename}.label",dtype=np.int32)
        # features = sp.csr_matrix(idx_features[:, 1:], dtype=np.float32)
        features = np.array(idx_features[:, 1:], dtype=np.float32)
        # print(features)

        # build graph
        idx = np.array(idx_features[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}

        adjs = None
        for i in range(self.view_size):
            edges_unordered = np.genfromtxt(f"{self.file_path}/{filename}_{i}.rel",
                                            dtype=np.int32)
            edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                             dtype=np.int32).reshape(edges_unordered.shape)
            # print(edges)
            ed1 = np.concatenate((edges[:, 0], edges[:, 1]),axis=0)
            ed2 = np.concatenate((edges[:, 1], edges[:, 0]),axis=0)
            # print(ed)
            adj = sp.coo_matrix((np.ones(edges.shape[0]*2), (ed1, ed2)),
                                shape=(len(idx), len(idx)),
                                dtype=np.float32)
            # print((edges[:, 0], edges[:, 1]))
            # print(adj)

            if not directed:
                # build symmetric adjacency matrix
                # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
                pass
            # features = normalize(features)
            adj = self.normalize(adj + sp.eye(adj.shape[0]))
            adj = self.sparse_mx_to_torch_sparse_tensor(adj)
            adj = adj.to_dense()
            # print(adj)
            adj = torch.unsqueeze(adj,0)
            # print(adj)
            if adjs is None:
                adjs = adj

            else:
                adjs = torch.cat((adjs,adj),0)

        # print(adjs)
        features = torch.FloatTensor(features)


        # return adj, features, labels, idx_train, idx_val, idx_test, and_children, or_children
        return adjs, features, y

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        mx = mx.dot(r_mat_inv)
        return mx

    # def encode_onehot(self, labels):
    #     classes = set(labels)
    #     classes_dict = {c: np.identity(int(max(classes))+1)[int(c), :] for i, c in
    #                     enumerate(classes)}
    #     labels_onehot = np.array(list(map(classes_dict.get, labels)),
    #                              dtype=np.int32)
    #     return labels_onehot

