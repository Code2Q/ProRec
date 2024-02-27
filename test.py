
from calendar import c
import pickle,torch
import numpy as np
# file = '/home/zoulixin/scratch/yq_code/rec/datasets/yelp2018/yelp2018_sparse_subgraph500_hop8_list.npz'
from torch_sparse.tensor import SparseTensor
import numpy as np
from model import *


file = '/home/zoulixin/scratch/yq_code/rec/checkpoints/ml-1M-pre-beauty-clamp-relu-Apool-4-epoch23-cl2.pth.tar'
file = '../checkpoints/outfile-Beauty-4-epoch2-cl2-prompt-Apool-tanh-relu.pth.tar'
# loaded_model = LightGCN()
# loaded_model.load_state_dict(torch.load('your_model.pth'))
# for name, param in loaded_model.named_parameters():
#     print(f"{name}: {param.data}")
# print(loaded_model.row_node.data)
state_dict = torch.load(file, map_location=torch.device('cpu'))

# Print the parameter names
print("Loaded Model Parameter Names:")
for name in state_dict:
    print(name)
print(type(state_dict['row_node']))
print(state_dict['row_node'].shape)
mask = state_dict['col_node'] > 0.7

# Use torch.nonzero to get indices where the mask is True
indices = torch.nonzero(mask)

# Print the indices
print("Indices of values greater than 0.5:")
print(indices)
print(indices.shape)
print(torch.max(state_dict['col_node'][indices]))


# result_np = np.tanh(0.15)
# print(result_np)
# result_np = np.tanh(0.5)
# print(result_np)
