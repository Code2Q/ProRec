import os
import utils
import torch
import numpy as np
import dataloader
import time
import Procedure
from parse import parse_args
import matplotlib.pyplot as plt
from os.path import join
from model import *

args = parse_args()
ROOT_PATH = "/home/zoulixin/scratch/yq_code/rec"
print(f"ROOT_PATH {ROOT_PATH}")
CODE_PATH = ROOT_PATH
CHECKPOINT_PATH = join(ROOT_PATH, 'checkpoints')
print(f"MODEL_PATH: {CHECKPOINT_PATH}")

if not os.path.exists(CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)



#set seed and device
utils.set_seed(args.seed)
print(">>SEED:", args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# load dataset
if args.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(args=args, path="../datasets/"+args.dataset)
elif args.dataset in [ 'Beauty',  'ml-1M']:
    dataset = dataloader.PointWise_Loader(args=args, path="../datasets/"+args.dataset)

# set model
model_name = args.model
MODELS = {
    # 'mf': PureMF,
    'lgn': LightGCN,
    # 'gcl': GCL_GT
}
Recmodel = MODELS[model_name](args, dataset)
Recmodel = Recmodel.to(device)
print(Recmodel)

total_params = 0


Neg_k = 1
bpr = utils.BPRLoss(Recmodel, args)
test_recall = []
recall_epoch = []
for epoch in range(args.epochs):
    start = time.time()
    # if epoch %10 == 0 and epoch > 0:
    print("[TEST]")
    results = Procedure.Test(dataset, Recmodel, epoch, device, args)
    test_recall.append(results)
    recall_epoch.append(epoch)
    output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, device=device)
    print(f'EPOCH[{epoch + 1}/{args.epochs}] {output_information}')
    file = f"{args.outfile}-{args.dataset}-{args.n_layer}-epoch{epoch}-cl{args.cl_layer}-Pretrain-token{args.token_num}-0105.pth.tar"
    weight_file = os.path.join(CHECKPOINT_PATH,file)
    torch.save(Recmodel.state_dict(), weight_file)
    print(f"weight saved path {weight_file}")
    # print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
    # torch.save(Recmodel.state_dict(), weight_file)

plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
plt.plot(recall_epoch, test_recall, 'bo-', label='Testing Recall')
plt.title('Testing Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.tight_layout()
plt.savefig('./recall_plot.jpg')
plt.show()
