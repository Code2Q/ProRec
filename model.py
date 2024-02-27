import torch
import dataloader
from dataloader import BasicDataset
import torch.nn.functional as F

from torch import nn
import numpy as np
from parse import parse_args
import time
from prompt import  *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.decomposition import PCA


args = parse_args()

def tanh_normalize(tensor):
    return 0.5 * (torch.tanh(tensor) + 1)

def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
    """
    Args:
        view1: (torch.Tensor - N x D)
        view2: (torch.Tensor - N x D)
        temperature: float
        b_cos (bool)

    Return: Average InfoNCE Loss
    """
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()

class AttentivePool(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(AttentivePool, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        attention_weights = F.softmax(self.linear(x), dim=1)
        pooled_output = torch.sum(x * attention_weights, dim=1)

        return pooled_output
    
class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class AE(torch.nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        layer_collector = []
        self.nu1 = 1e-8
        self.nu2 = 1e-5
        
        if args.prompt:
            layer_collector.append(torch.nn.Linear(500+args.token_num,128))
    
        else: layer_collector.append(torch.nn.Linear(500, 128))
        self.encoder = torch.nn.Sequential(*layer_collector)

        layer_collector1 = []
        if args.prompt:
            layer_collector1.append(torch.nn.Linear(128,500+args.token_num))
            layer_collector1.append(torch.nn.LeakyReLU())
        else: layer_collector1.append(torch.nn.Linear(128, 500))
        self.decoder = torch.nn.Sequential(*layer_collector1)
        self.layer_collector = layer_collector + layer_collector1

    def forward(self, x): #x: bs,input_size

        feat = self.encoder(x) #feat: bs,latent_size
        re_x = self.decoder(feat) #re_x: bs, output_size
        return feat, re_x
    
    @staticmethod
    def _L_1st(a, e_n, e):
        result = e_n - 2 * torch.matmul(e, e.transpose(1, 2)) + e_n.transpose(1,2)
        return (a * result).sum(dim=(0, 1, 2))
        # return (a * (e_n - 2 * torch.mm(e, e.t()) + e_n.t())).sum()

    @staticmethod
    def _L_2nd(a_b, f, b):
        x1 = (((a_b - f) * b) ** 2).sum()
        return x1

    def loss(self, a_b, b, embeddings, final):
        # L, l2 = 0, 0
        embeddings_norm = (embeddings ** 2).sum(2, keepdims=True)
        l1 = 20 * self._L_1st(a_b, embeddings_norm, embeddings) 
        l2 =  self._L_2nd(a_b, final, b)
        L = args.sdne_rate * (l2 + l1)
        for param in self.layer_collector:
            if type(param) == torch.nn.Linear:
                # print(f"loss l1{l1} l2{l2} linear_norm{(param.weight ** 2).sum()} {param.weight.abs().sum()}")
                L += self.nu2 * (param.weight ** 2).sum() + self.nu1 * param.weight.abs().sum()
        return L, l2, l1
    
class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = 128
        self.edge_index = self.dataset.edge_index_symmetric
        self.n_layers = self.config.n_layer
        self.keep_prob = self.config.keepprob
        self.subgraph = self.dataset.get_subgraph()
        self.SDNE = AE()
        self.eps = float(0.2)
        self.cl_rate = 0.2
        # self.temp = 0.15
        self.temp = args.tmp
        self.layercl = self.config.cl_layer
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.use_attentive =False
        # self.act = nn.Sigmoid()

        if self.use_attentive:
            self.attentive_pool = AttentivePool(128,1)
        if self.config.prompt:
            self.row_node = torch.nn.Parameter(torch.empty(args.token_num, 500).cuda().half())
            self.col_node = torch.nn.Parameter(torch.empty(500+args.token_num, args.token_num).cuda().half())
            
            # random_indices = torch.randperm(510)[:50]
            # random_indices = torch.randint(0, 500, (10,50))
            # mask = torch.zeros_like(parameter_tensor)
            # mask.scatter_(1, random_indices, 1) 
            # self.row_node.data = mask
            # self.row_node[random_indices].data = torch.ones_like(random_indices).float()

            torch.nn.init.uniform_(self.row_node, a=0, b=1)
            torch.nn.init.uniform_(self.col_node, a=0, b=1)
            # self.col_node.data = mask
            # self.col_node[random_indices].data = torch.ones_like(random_indices).float()
            # self.row_node.data = torch.clamp(self.row_node.data,0,1)
            # self.col_node.data = torch.clamp(self.col_node.data,0,1)
    

        if self.config.pretrain == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            print('use NORMAL distribution initilizer')

        # self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        # self.act = torch.nn.LeakyReLU()
        print(f"lgn is already to go(dropout:{self.config.dropout})")

    def __dropout_x(self, x, keep_prob):

        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def cal_cl_loss(self, idx, user_view1,user_view2,item_view1,item_view2):
        user_cl_loss = InfoNCE(user_view1, user_view2, self.temp)
        item_cl_loss = InfoNCE(item_view1, item_view2, self.temp)
        return user_cl_loss + item_cl_loss
    
    def computer(self, perturbed = False):
        ego_embeddings = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        all_embeddings_cl  = ego_embeddings

        all_embs = []
        if self.config.dropout:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(g_droped, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embs.append(ego_embeddings)
            if layer == self.layercl - 1:
                all_embeddings_cl = ego_embeddings
        final_embs = torch.stack(all_embs, dim=1)
        final_embs = torch.mean(final_embs, dim=1)
        users, items = torch.split(final_embs, [self.num_users, self.num_items])
        users_cl, items_cl = torch.split(all_embeddings_cl, [self.num_users, self.num_items])
        if perturbed:
            return users, items, users_cl, items_cl
        return users, items
    
    def getUsersRating(self, users, items, b):

        item_add = items + self.num_users
        startu, endu = users * 500, users * 500 + 500
        self.row_node.data = torch.clamp(self.row_node.data,0,1)
        self.col_node.data = torch.clamp(self.col_node.data,0,1)

        g = self.subgraph.cuda()

        graph_listu = [g[startu[i]:endu[i], startu[i]:endu[i]].to_dense() for i in range(len(users))]
        users_g = torch.stack(graph_listu).cuda()

        if args.prompt:
            users_g = torch.cat([users_g,self.row_node.unsqueeze(0).expand(users_g.shape[0],-1,-1)], dim=1)
            users_g = torch.cat([users_g,self.col_node.unsqueeze(0).expand(users_g.shape[0],-1,-1)], dim=2)

        users_sdne, _ = self.SDNE(users_g)
        del users_g


        if self.use_attentive:
            users_sdne = self.attentive_pool(users_sdne)
        else:   
            users_sdne = torch.mean(users_sdne, dim=1, keepdim=False)
            

        starti, endi = item_add * 500, item_add * 500 + 500
        graph_listi = [g[starti[i]:endi[i], starti[i]:endi[i]].to_dense() for i in range(len(items))]
        item_g = torch.stack(graph_listi).cuda()
        if args.prompt:
            item_g = torch.cat([item_g,self.row_node.unsqueeze(0).expand(item_g.shape[0],-1,-1)], dim=1)
            item_g = torch.cat([item_g,self.col_node.unsqueeze(0).expand(item_g.shape[0],-1,-1)], dim=2)

        item_sdne, _ = self.SDNE(item_g)
        if self.use_attentive:
            item_sdne = self.attentive_pool(item_sdne) 
        else: item_sdne = torch.mean(item_sdne, dim=1, keepdim=False)
        del item_g

        users_emb = users_sdne
        items_emb = item_sdne

        rating = torch.matmul(users_emb, items_emb.t())

        return rating
    
    def getEmbedding(self, users, pos_items, neg_items, perturbed):
     
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        if perturbed:
            all_users, all_items, all_users_cl, all_items_cl = self.computer(perturbed)
            return all_users, all_items, users_emb_ego, pos_emb_ego, neg_emb_ego, all_users_cl, all_items_cl
        else:
            all_users, all_items= self.computer(perturbed)
            return all_users, all_items, users_emb_ego, pos_emb_ego, neg_emb_ego
        
     
    def bpr_loss(self, users, pos, neg):

        item_adduser_pos = pos + self.num_users
        item_adduser_neg = neg + self.num_users

        subg_idx = torch.cat((users,item_adduser_pos,item_adduser_neg),dim=0)
        graph_list = []
        start, end = subg_idx * 500, subg_idx * 500 + 500
        g = self.subgraph.cuda()

        graph_list = [g[start[i]:end[i], start[i]:end[i]].to_dense() for i in range(len(subg_idx))]

        sub_g = torch.stack(graph_list).cuda() #torch.Size([4096, 500, 500])
        # print(self.row_node.grad, self.row_node.is_leaf)

        if args.prompt:
            self.row_node.data = torch.clamp(self.row_node.data,0,1)
            self.col_node.data = torch.clamp(self.col_node.data,0,1)
            
            sub_g = torch.cat([sub_g,self.row_node.unsqueeze(0).expand(sub_g.shape[0],-1,-1)], dim=1)
            sub_g = torch.cat([sub_g,self.col_node.unsqueeze(0).expand(sub_g.shape[0],-1,-1)], dim=2)


        
        b_mat_train = torch.ones_like(sub_g)
        b_mat_train[sub_g != 0] = 5.
        user_num = users.size()[0]

        feat, final = self.SDNE(sub_g)
        
        sdne_loss, l2, l1 = self.SDNE.loss(sub_g, b_mat_train, feat, final)
        del sub_g
        if self.use_attentive:
            mean_node = self.attentive_pool(feat)
        else: mean_node = torch.mean(feat, dim=1, keepdim=False)

        perturbed = True

        if perturbed:
            (all_users, all_items, userEmb0,  posEmb0, negEmb0, users_cl, items_cl) = self.getEmbedding(users.long(), pos.long(), neg.long(), perturbed)
            cl_loss = self.cl_rate * self.cal_cl_loss([users,pos],all_users,users_cl,all_items,items_cl)
    
        else:
            all_users, all_items, userEmb0,  posEmb0, negEmb0 = self.getEmbedding(users.long(), pos.long(), neg.long(), perturbed) 
        user_num = users.size()[0]
        pos_num = pos.size()[0]


        batch_users_emb = all_users[users] + mean_node[ : user_num, : ]
        batch_pos_emb = all_items[pos] + mean_node[ user_num : user_num  + pos_num, : ]
        batch_neg_emb = all_items[neg] + mean_node[ user_num + pos_num :, :]

        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))

        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        print(f"brp loss {loss}")

        return loss, reg_loss, sdne_loss, l2, l1, cl_loss
       