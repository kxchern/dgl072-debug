
import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ogb.graphproppred.dataset_dgl import DglGraphPropPredDataset, collate_dgl

from optims.optim_auc_with_bce import ModelOptLearning_BCE
from optims.optim_auc_with_cls import ModelOptLearning_CLS
from models.GIN import GIN
from models.GCN import GCN


### load dataset 
def load_data(args):    

    dataset = DglGraphPropPredDataset(name=args.dataset, root=args.datadir)
    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, 
                              collate_fn=collate_dgl, num_workers=4)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, 
                              collate_fn=collate_dgl, num_workers=4)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, 
                              collate_fn=collate_dgl, num_workers=4)    
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
    if args.loss_type == 'bce':
        modelOptm = ModelOptLearning_BCE(
                                model=model, 
                                optimizer=optimizer,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                test_loader=test_loader,
                                args=args)
    elif args.loss_type == 'cls': 
        modelOptm = ModelOptLearning_CLS(
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
    args.best_dir = os.path.join(args.logs_dir, 'best')
    return args

def print_best_log(args, key_metric='valid-rocauc', eopch_slice=0, 
                   metric_list='all'):
    logs_table = pd.read_excel(os.path.join(args.xlsx_dir, args.identity+'.xlsx'))
    metric_log = logs_table[key_metric]
    best_epoch = metric_log[eopch_slice:].idxmax()
    best_frame = logs_table.loc[best_epoch]
    if not os.path.exists(args.best_dir):
        os.mkdir((args.best_dir))
    best_frame.to_excel(os.path.join(args.best_dir, args.identity+'.xlsx'))

    if metric_list == 'all':
        print(best_frame)
    else:
        for metric in metric_list:
            print(f'{metric }: {best_frame[metric]}')
    return 0

def plot_logs(args, metric_list='all'):
    if not os.path.exists(args.imgs_dir):
        os.mkdir(args.imgs_dir)
    logs_table = pd.read_excel(os.path.join(args.xlsx_dir, args.identity+'.xlsx'))

    for metric in metric_list:
        metric_epochs = logs_table[metric]
        plt.plot(range(len(metric_epochs)), metric_epochs, 'g-')
        plt.savefig(os.path.join(args.imgs_dir, args.identity + f'-{metric}.png'))
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


