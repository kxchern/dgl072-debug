
import torch
import argparse, os
import torch.nn as nn
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

from utils.utils_mask import *
from utils.grad_constraint_with_cls import HookModule

dir_path = os.path.dirname(__file__)

criterion = nn.CrossEntropyLoss()


def main(args):
    dataset, train_loader, valid_loader, test_loader = load_data(args)
    torch.cuda.set_device(0)
    args = args_(args, dataset)
    set_seed(args)

    model = load_model(args, dataset)
    model.eval()

    module = HookModule(model=model, module=model.conv_layers[3])   
    pth_path = os.path.join(args.dict_dir, args.identity+'.pth')
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.to(args.device)


    for graphs, labels in train_loader:
        graphs, labels = graphs.to(args.device),labels.to(args.device)
        nfeats = graphs.ndata['feat']
        efeats = graphs.edata['feat']

        out = model(graphs, nfeats, efeats)
        out_, indx_ = out.sort(dim = -1, descending=True)
        batch_num_nodes = graphs.batch_num_nodes()
        # module = HookModule(model=model, module=model.conv_layers[3])   
        for i in range(graphs.batch_size):
            pred_score = out_[i][0]
            grads = module.grads(outputs = pred_score, inputs = module.activations)[sum(batch_num_nodes[0:i]):sum(batch_num_nodes[0:i+1]),:]
            print(grads)

    return 0

if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--datadir", type=str, default='dataset')
    parser.add_argument("--dataset", type=str, default='ogbg-molbbbp')

    parser.add_argument("--model", type=str, default='GCN', choices='GIN, GCN')
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--epoch_slice", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--loss_type", type=str, default='cls', choices='bce, cls')
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--logs_dir", type=str, default=os.path.join(dir_path,'logs'))
    parser.add_argument("--mask_dir", type=str, default=os.path.join(dir_path,'mask'))
    parser.add_argument("--mask_cls_dir", type=str, default=os.path.join(dir_path,'mask_cls'))

    args = parser.parse_args()
    
    main(args)



