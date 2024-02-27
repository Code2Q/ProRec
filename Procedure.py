
import numpy as np
import torch
from torch import nn
import utils
from pprint import pprint
from parse import parse_args
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score

CORES = multiprocessing.cpu_count() // 2

args = parse_args()
topks = eval(args.topks)
def BPR_train_original(dataset, recommend_model, loss_class, epoch, device, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(device)
    posItems = posItems.to(device)
    negItems = negItems.to(device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // args.bpr_batch + 1
    print(f"total batch {total_batch}")
    aver_loss = 0.

    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=args.bpr_batch)):
        
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        # print(f"batch {batch_i} cri {cri}")
        aver_loss += cri
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def Test(dataset, Recmodel, epoch, device, w=None, multicore=0):
    u_batch_size = args.testbatch
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.GCL_GT
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(topks)),
               'recall': np.zeros(len(topks)),
               'ndcg': np.zeros(len(topks))}
    with torch.no_grad():
        sigmoid  = nn.Sigmoid()
        users = list(testDict.keys())
        items = list(range(dataset.m_items))
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        
        total_batch = len(users) // u_batch_size + 1
        i=0
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(device)
            ratings = []
            
            for batch_items in utils.minibatch(items, batch_size = u_batch_size):
                batch_items_gpu = torch.Tensor(batch_items).long()
                batch_items_gpu = batch_items_gpu.to(device)
                rating = Recmodel.getUsersRating(batch_users_gpu, batch_items_gpu, i)
                i+=1
                ratings.append(rating)
                # rating = rating.cpu()
            ratings_all = torch.cat(ratings, dim=1)
            print(f"ratings  {ratings_all.shape}")
            # ratings = torch.tensor(ratings, dtype=torch.float32)
            rating = sigmoid(ratings_all)
            del ratings
            exclude_index = []
            exclude_items = []
            for range_i, item in enumerate(allPos):
                exclude_index.extend([range_i] * len(item))
                exclude_items.extend(item)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)

        if multicore == 1:
            pool.close()
        print(results)
        return results