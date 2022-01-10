
import torch 
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from dgl.nn.pytorch.glob import AvgPooling, SumPooling, MaxPooling

from models.norm import Norm

class GCNConvLayer(nn.Module):
    def __init__(self, embed_dim, aggregator_type='mean'):
        super(GCNConvLayer, self).__init__()

        self.update_feat = nn.Linear(embed_dim, embed_dim)
        self.graphs_norm = Norm('gn', embed_dim)

        if aggregator_type == 'sum':
            self.reduce = fn.sum
        elif aggregator_type == 'mean':
            self.reduce = fn.mean
        elif aggregator_type == 'max':
            self.reduce = fn.max
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))


    def forward(self, graphs, nfeat, efeat):
        graphs = graphs.local_var()

        graphs.ndata['h_n'] = nfeat
        graphs.edata['h_e'] = efeat
        graphs.update_all(fn.u_add_e('h_n', 'h_e', 'm'),
                          self.reduce('m', 'neigh'))
        rst = self.update_feat(graphs.ndata['neigh'] +  nfeat)

        # # aggragation
        # graphs.ndata['feat'] = nfeat
        # graphs.apply_edges(fn.copy_u('feat', 'e'))
        # # graphs.edata['e'] = F.relu(efeat + graphs.edata['e'])
        # graphs.edata['e'] = efeat + graphs.edata['e']
        # graphs.update_all(fn.copy_e('e','m'), self.reduce('m', 'feat'))

        # # node feature updating 
        # rst = self.update_feat(graphs.ndata['feat'] + nfeat)

        # graphs norm in batch graphs
        rst = self.graphs_norm(graphs, rst)
        return rst


class GCN(nn.Module):
    def __init__(self, embed_dim, output_dim, num_layer,
                       aggregator_type='mean', pooling_type="mean", 
                       activation=F.relu, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layer = num_layer
        # input layer
        self.atom_encoder = AtomEncoder(embed_dim)
        # convolutional layer & bond layer 
        self.bond_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for i in range(num_layer-1):
            self.bond_layers.append(BondEncoder(embed_dim))
            self.conv_layers.append(GCNConvLayer(embed_dim))
        # output layer
        self.predict = nn.Linear(embed_dim, output_dim)     

        # modules in GNN
        if pooling_type == "sum":
            self.pooling = SumPooling()
        elif pooling_type == "mean":
            self.pooling = AvgPooling()
        elif pooling_type == "max":
            self.pooling = MaxPooling()
        else:
            raise KeyError('Pooling type {} not recognized.'.format(pooling_type))
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, graphs, nfeat, efeat):
        # initializing node features h_n
        h_n = self.atom_encoder(nfeat)
        h_list = [h_n]
        for layer in range(self.num_layer-1):
            x = h_n
            # initializing edge features h_e & graph convolution for node features 
            h_e = self.bond_layers[layer](efeat)
            h_n = self.conv_layers[layer](graphs, h_n, h_e)
            # activation & residual 
            if layer != self.num_layer - 2:
                h_n = self.activation(h_n)
            h_n = h_n + x                   
        # pooling & prediction
        g_n = self.pooling(graphs, h_n)
        pre = self.dropout(self.predict(g_n))
        return pre
