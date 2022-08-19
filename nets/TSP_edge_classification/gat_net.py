import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_layer import GATLayer, CustomGATLayer, CustomGATLayerEdgeReprFeat, CustomGATLayerIsotropic
from layers.mlp_readout_layer import MLPReadout

class GATNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
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
        self.n_classes = n_classes
        self.device = net_params['device']
        
        self.layer_type = {
            "dgl": GATLayer,
            "edgereprfeat": CustomGATLayerEdgeReprFeat,
            "edgefeat": CustomGATLayer,
            "isotropic": CustomGATLayerIsotropic,
        }.get(net_params['layer_type'], GATLayer)
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim * num_heads)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        #self.layers = nn.ModuleList([self.layer_type(hidden_dim * num_heads, hidden_dim, num_heads,
        #                                             dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        
        
        layers_list = []
        for i in range(n_layers-1):
            if i in [0, 2]: #[0]
                edge_lr = True
            else:
                edge_lr = False
            layers_list.append(CustomGATLayerEdgeReprFeat(hidden_dim * num_heads, hidden_dim, num_heads,
                                              dropout, self.batch_norm, self.residual, edge_lr))
        layers_list.append(CustomGATLayerEdgeReprFeat(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual, edge_lr=False))
        self.layers = nn.ModuleList(layers_list)
        
        self.MLP_layer = MLPReadout(3*out_dim, n_classes)

        self.merg = MERG(in_dim, hidden_dim, num_heads)
                
    def forward(self, g, h, e):
        
        if not self.edge_feat:
            e = torch.ones_like(e).to(self.device)
        e = self.embedding_e(e.float())
        h = self.embedding_h(h.float())
        h = self.in_feat_dropout(h)
        
        for conv in self.layers:
            h, e = conv(g, h, e)
        
        g.ndata['h'] = h
        

        def _edge_feat(edges):
            #import pdb;pdb.set_trace()
            #e = torch.cat([edges.src['h'], edges.dst['h'],self.lr_e], dim=1)
            #e = self.MLP_layer(e)
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            return {'e': e}
        g.apply_edges(_edge_feat)
        
        lr_e = torch.cat([g.edata['e'], e], dim=1)
        lr_e = self.MLP_layer(lr_e)
        return lr_e
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(weight=None)
        loss = criterion(pred, label)

        return loss
