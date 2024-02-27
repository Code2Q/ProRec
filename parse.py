
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=1024,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--outfile', type=str, default="outfile",
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--cl_layer', type=int,default=2,
                        help="the embedding size of lightGCN")
    parser.add_argument('--input_dim_rec', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--n_layer', type=int,default=4,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-6,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=1,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.5,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=3000,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='gowalla',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book, Beauty]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[10,20,40]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model')
    parser.add_argument('--A_split', type=bool, default=False)
    parser.add_argument('--prompt', type=bool,default=True,
                        help="whether to use prompt tuning")
    parser.add_argument('--token_num', type=int,default=7)
    parser.add_argument('--sdne_rate', type=float,default=1e-6)
    parser.add_argument('--tmp', type=float,default=0.2)
    return parser.parse_args()
