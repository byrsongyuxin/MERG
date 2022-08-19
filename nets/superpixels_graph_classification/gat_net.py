import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_layer import GATLayer, CustomGATLayerEdgeReprFeat, CustomGATHeadLayerEdgeReprFeat
from layers.mlp_readout_layer import MLPReadout

class GATNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        num_heads = net_params['n_heads']

        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.dropout = dropout
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim * num_heads)
        in_dim_edge = net_params['in_dim_edge']
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim * num_heads)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
#        self.layers = nn.ModuleList([GATLayer(hidden_dim * num_heads, hidden_dim, num_heads,
#                                              dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
#        self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual))
        
        layers_list = []
        for i in range(n_layers-1):
            if i in [0, ]: #[0, 2]
                edge_lr = True
            else:
                edge_lr = False
            layers_list.append(CustomGATLayerEdgeReprFeat(hidden_dim * num_heads, hidden_dim, num_heads,
                                              dropout, self.batch_norm, self.residual, edge_lr))
        layers_list.append(CustomGATLayerEdgeReprFeat(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual, edge_lr=False))
        self.layers = nn.ModuleList(layers_list)
        
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        
    def forward(self, g, h, e):
        h = self.embedding_h(h)
        e = self.embedding_e(e)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h,e = conv(g, h, e)
        g.ndata['h'] = h
        
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        return self.MLP_layer(hg)
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss
