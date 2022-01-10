
import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ogb.graphproppred.dataset_dgl import DglGraphPropPredDataset, collate_dgl
from typing import Optional, Tuple
from torch import Tensor 

from optims.modelsecond_optim_auc_with_cls import ModelSecondOptLearning_CLS
from models.GIN import GIN
from models.GCN import GCN


### load dataset 
def load_data(args):    

    dataset = DglGraphPropPredDataset(name=args.dataset, root=args.datadir)
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate_dgl, num_workers=0)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, 
                              collate_fn=collate_dgl, num_workers=0)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, 
                              collate_fn=collate_dgl, num_workers=0)    
    print('- ' * 30)
    print(f'{args.dataset} dataset loaded.')
    print('- ' * 30)
    return dataset, train_loader, valid_loader, test_loader


### load gnn model
def load_model(args, dataset):
    if args.model == 'GCN':
        model = GCN(args.embed_dim, args.output_dim, args.num_layer).to(args.device)
    elif args.model == 'GIN':
        model = GIN(args.embed_dim, args.output_dim, args.num_layer).to(args.device)
    print('- ' * 30)
    print(f'{args.model} neural network loaded.')
    print('- ' * 30)   
    return model


### load model optimizing and learning class
def ModelOptLoading(model, optimizer, 
                    train_loader, valid_loader, test_loader,
                    args):
    if args.loss_type == 'cls': 
        modelOptm = ModelSecondOptLearning_CLS(
                                model=model, 
                                optimizer=optimizer,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                test_loader=test_loader,
                                args=args)
    return modelOptm


### add arguments
def args_(args, dataset): 
    if args.loss_type == 'bce':
        args.output_dim = dataset.num_tasks
    elif args.loss_type == 'cls': 
        args.output_dim = int(dataset.num_classes)
    args.inputE_dim = dataset[0][0].edata['feat'].shape[-1]
    args.identity = (f"{args.dataset}-"+
                     f"{args.model}-"+
                    #  f"{args.epochs}-"+
                     f"{args.batch_size}-"+
                     f"{args.num_layer}-"+
                     f"{args.embed_dim}-"+
                     f"{args.lr}-"+
                     f"{args.dropout}-"+
                     f"{args.weight_decay}-"+
                     f"{args.loss_type}-"+
                     f"{args.seed}"
                     )
    if not os.path.exists(args.logs_dir):
        os.mkdir(args.logs_dir)
    args.xlsx_dir = os.path.join(args.logs_dir, 'xlsx')
    args.imgs_dir = os.path.join(args.logs_dir, 'imgs')
    args.dict_dir = os.path.join(args.logs_dir, 'dict')
    # second training logs dir
    args.second_identity = (f'{args.identity}-{args.runs}')
    args.second_logs_dir = 'logs_second'
    if not os.path.exists(args.second_logs_dir):
        os.mkdir(args.second_logs_dir)
    args.second_xlsx_dir = os.path.join(args.second_logs_dir, 'xlsx')
    args.second_imgs_dir = os.path.join(args.second_logs_dir, 'imgs')
    args.second_dict_dir = os.path.join(args.second_logs_dir, 'dict')
    args.second_best_dir = os.path.join(args.second_logs_dir, 'best')
    return args

def print_best_log(args, key_metric='valid-rocauc', eopch_slice=0, 
                   metric_list='all'):
    logs_table = pd.read_excel(os.path.join(args.second_xlsx_dir, args.second_identity+'.xlsx'))
    metric_log = logs_table[key_metric]
    best_epoch = metric_log[eopch_slice:].idxmax()
    best_frame = logs_table.loc[best_epoch]
    if not os.path.exists(args.second_best_dir):
        os.mkdir((args.second_best_dir))
    best_frame.to_excel(os.path.join(args.second_best_dir, args.second_identity+'.xlsx'))

    if metric_list == 'all':
        print(best_frame)
    else:
        for metric in metric_list:
            print(f'{metric }: {best_frame[metric]}')
    return 0

def plot_logs(args, metric_list='all'):
    if not os.path.exists(args.second_imgs_dir):
        os.mkdir(args.second_imgs_dir)
    logs_table = pd.read_excel(os.path.join(args.second_xlsx_dir, args.second_identity+'.xlsx'))

    for metric in metric_list:
        metric_epochs = logs_table[metric]
        plt.plot(range(len(metric_epochs)), metric_epochs, 'g-')
        plt.savefig(os.path.join(args.second_imgs_dir, args.second_identity + f'-{metric}.png'))
        plt.close()
    return 0 


### set random seed
def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if args.device >= 0:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)



def my_to_dense_batch(num_nodes, x: Tensor, batch: Optional[Tensor] = None,
                   fill_value: float = 0., max_num_nodes: Optional[int] = None,
                   batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    r"""Given a sparse batch of node features
    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}` (with
    :math:`N_i` indicating the number of nodes in graph :math:`i`), creates a
    dense node feature tensor
    :math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}` (with
    :math:`N_{\max} = \max_i^B N_i`).
    In addition, a mask of shape :math:`\mathbf{M} \in \{ 0, 1 \}^{B \times
    N_{\max}}` is returned, holding information about the existence of
    fake-nodes in the dense representation.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered. (default: :obj:`None`)
        fill_value (float, optional): The value for invalid entries in the
            resulting dense output tensor. (default: :obj:`0`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)
        batch_size (int, optional) The batch size. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`BoolTensor`)
    """
    if batch is None and max_num_nodes is None:
        mask = torch.ones(1, x.size(0), dtype=torch.bool, device=x.device)
        return x.unsqueeze(0), mask

    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    # num_nodes = torch.tensor(num_nodes).to(x.device)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())

    idx = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)

    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    out = x.new_full(size, fill_value)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool,
                       device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return out, mask, num_nodes


