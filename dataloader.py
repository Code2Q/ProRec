

import os
from os.path import join
import sys,pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix, load_npz
import scipy.sparse as sp
import networkx as nx
from time import time
from torch_sparse.tensor import SparseTensor

print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print("CUDA is available")
    print("CUDA version:", torch.version.cuda)
    device = torch.device("cuda")
else:
    print("CUDA is not available")
    device = torch.device("cpu")

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError
    def getGraphEmbedding(self):
        raise NotImplementedError

def is_valid_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False



class Loader(BasicDataset):
    """
    gowalla dataset
    """

    def __init__(self, args, path="../datasets/yelp2018"):
        # train or test
        print(f'loading [{path}]')
       
        # self.folds = args.a_fold
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.dataname = args.dataset

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if (len(l) > 2 or is_valid_integer(l[1])):
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        testUniqueUsers.append(uid)
                        testUser.extend([uid] * len(items))
                        testItem.extend(items)
                        self.m_item = max(self.m_item, max(items))
                        self.n_user = max(self.n_user, uid)
                        self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        trainUser = np.array(self.trainUser)
        trainItem = np.array(self.trainItem+self.n_user)
        # print(trainItem.size())
        # print(trainUser.size())
        combined_array = np.stack((trainUser, trainItem), axis=0)
        # print(combined_array.size)
        self.edge_index = torch.tensor(combined_array, dtype=torch.long)
        self.edge_index_symmetric = torch.cat([self.edge_index, self.edge_index.flip(0)], dim=1).to(device)
        print(f"===edge index shape {self.edge_index_symmetric.size()}")

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{args.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")


        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        #计算了每个用户对应行的非零元素总和，即该用户与多少个物品发生了交互。
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        #这将用户的度中为零的值（表示没有交互）替换为 1，以避免分母为零的情况。
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        #获取所有用户的正向交互物品列表
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{args.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float16)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float16)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)


            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(device)
            print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def get_subgraph(self):

        file ='/home/zoulixin/scratch/yq_code/rec/datasets/{}/{}_sparse_subgraph500_hop8_coo.npz'.format(self.dataname, self.dataname) 
        coo = load_npz(file)
        if self.dataname == 'ml-1M':
            row = torch.Tensor(coo.row).long()
            col = torch.Tensor(coo.col).long()
        else:
            row = torch.Tensor(coo.row).long() - 1
            col = torch.Tensor(coo.col).long() - 1
        # min_col = torch.min(col)
        # min_row = torch.min(row)
        # max_col = torch.max(col)
        # max_row = torch.max(row)
        # print(f"coo shape {coo.shape}")
        # print(f"max {max_col.item()} {max_row.item()} min {min_col.item()} {min_row.item()}")
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        print(index.shape)
        print(data.shape)
        print(coo.shape)
        fullg_coo = torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))
        fullg = SparseTensor.from_torch_sparse_coo_tensor(fullg_coo)
        print(type(fullg))
        return fullg
        


class PointWise_Loader(BasicDataset):
    def __init__(self, args, path="/home/zoulixin/scratch/yq_code/rec/datasets/Beauty"):
        print(f'loading [{path}]')
        # self.split = args.A_split
        # self.folds = args.a_fold
        self.Graph = None
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainFullUsers, trainItem, trainFullItems, trainUser = [], [], [], [], []
        testUniqueUsers, testFullUsers, testItem, testFullItems, testUser = [], [], [], [], []
        self.traindataSize = 0
        self.testdataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 1:
                    if args.dataset == 'Beauty':
                        l = l.strip('\n').split(',')
                        item_id = int(l[1]) - 1
                        user_id = int(l[0]) - 1
                    else:
                        l = l.strip('\n').split(' ')
                        item_id = int(l[1])-1
                        user_id = int(l[0])-1
                    trainUniqueUsers.append(user_id)
                    trainUser.append(user_id)
                    trainItem.append(item_id)
                    self.m_item = max(self.m_item, item_id)
                    self.n_user = max(self.n_user, user_id)
                    self.traindataSize += 1
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.trainUniqueUsers = np.array(set(trainFullUsers))
        self.trainUniqueItems = np.array(set(trainFullItems))

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 1:
                    if args.dataset == 'Beauty':
                        l = l.strip('\n').split(',')
                    else:
                        l = l.strip('\n').split(' ')
                    item_id = int(l[1])-1
                    user_id = int(l[0])-1
                    testFullUsers.append(user_id)
                    testFullItems.append(item_id)
                    self.m_item = max(self.m_item, item_id)
                    self.n_user = max(self.n_user, user_id)
                    self.testdataSize += 1
        self.m_item += 1
        self.n_user += 1

        self.testFullUsers = np.array(testFullUsers)
        self.testUniqueUsers = np.array(set(testFullUsers))

        self.testFullItems = np.array(testFullItems)
        self.testUniqueItems = np.array(set(testFullItems))

        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"Beauty Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")
        # print(f"len(self.trainFullUsers) {len(self.trainFullUsers)}")


        trainUser = np.array(self.trainUser)
        trainItem = np.array(self.trainItem+self.n_user)
        combined_array = np.stack((trainUser, trainItem), axis=0)
        # print(combined_array.size)
        self.dataname = "Beauty"
        self.edge_index = torch.tensor(combined_array, dtype=torch.long)
        self.edge_index_symmetric = torch.cat([self.edge_index, self.edge_index.flip(0)], dim=1).to(device)

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_users, self.m_items))
 
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()

        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        # self.Graph = self.getNetworkxGraph()
        self.Graph = self.getSparseGraph()
        
        print(f" networkx graph is loaded : {self.Graph}")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDataSize(self):
        return self.testdataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float16)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
    def getNetworkxGraph(self):
        Graph = nx.Graph()
        print("loading graph by networkx")
        item_idx_add = self.n_users
        print(f"users: {self.n_users} item : {self.m_items}")
        for i in range(self.trainDataSize):
            Graph.add_edge(self.trainFullUsers[i], self.trainFullItems[i]+item_idx_add)
            Graph.add_edge(self.trainFullItems[i]+item_idx_add, self.trainFullUsers[i])
        for i in range(self.testDataSize):
            Graph.add_edge(self.testFullUsers[i], self.testFullItems[i]+item_idx_add)
            Graph.add_edge(self.testFullItems[i]+item_idx_add, self.testFullUsers[i])
        return Graph


    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
                norm_adj = norm_adj.tocoo()
                row = torch.from_numpy(norm_adj.row).to(torch.long)
                col = torch.from_numpy(norm_adj.col).to(torch.long)
                # edge_index = torch.stack([row, col], dim=0).to(device)
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float16)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                #norm_adj = norm_adj.tocsr()

                norm_adj = norm_adj.tocoo()

                row = torch.from_numpy(norm_adj.row).to(torch.long)
                col = torch.from_numpy(norm_adj.col).to(torch.long)
                # edge_index = torch.stack([row, col], dim=0).to(device)

                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            # if self.split == True:
            #     self.Graph = self._split_A_hat(norm_adj)
            #     print("done split matrix")
            # else:
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(device)
            print("don't split the matrix")
        return self.Graph
        # return edge_index


    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testFullItems):
            user = self.testFullUsers[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    # def getUserItemFeedback(self, users, items):
    #     # print(self.UserItemNet[users, items])
    #     return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def get_subgraph(self):
        if self.dataname == 'Beauty':
            file ='/home/zoulixin/scratch/yq_code/rec/datasets/{}/{}_sparse_subgraph500_hop10_coo.npz'.format(self.dataname, self.dataname) 
        else: file = '/home/zoulixin/scratch/yq_code/rec/datasets/{}/{}_sparse_subgraph500_hop8_coo.npz'.format(self.dataname, self.dataname) 
        coo = load_npz(file)

        row = torch.Tensor(coo.row).long() - 1
        col = torch.Tensor(coo.col).long() - 1
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        fullg_coo = torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))
        fullg = SparseTensor.from_torch_sparse_coo_tensor(fullg_coo)
        print(type(fullg))
        return fullg

def main():
    beauty_loader = PointWise_Loader()
    dataset = beauty_loader
    print(dataset.n_users)





if __name__ == "__main__":
    main()
