import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.gated_gcn_layer import GatedGCNLayer, GatedGCNLayerEdgeFeatOnly, GatedGCNLayerIsotropic
from layers.mlp_readout_layer import MLPReadout

class MERG(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        self.proj1 = nn.Linear(in_dim,hidden_dim**2) 
        self.proj2 = nn.Linear(in_dim,hidden_dim) 
        self.edge_proj = nn.Conv1d(in_channels=2,out_channels=1,kernel_size=3,padding=1) 
        self.edge_proj2 = nn.Linear(in_dim,hidden_dim) 
        self.edge_proj3 = nn.Linear(hidden_dim,hidden_dim) 
        self.hidden_dim = hidden_dim 
        
        self.bn_node_lr_e = nn.BatchNorm1d(hidden_dim)

    def forward(self, g, h, e):
        g.apply_edges(lambda edges: {'src' : edges.src['feat']})
        src = g.edata['src'].unsqueeze(1) #[M,1,D]
        g.apply_edges(lambda edges: {'dst' : edges.dst['feat']})
        dst = g.edata['dst'].unsqueeze(1) #[M,1,D]
        edge = torch.cat((src,dst),1).to(h.device) #[M,2,D]
        lr_e_local = self.edge_proj(edge).squeeze(1)#[M,D]
        lr_e_local = self.edge_proj2(lr_e_local)
        N = h.shape[0]
        h_proj1 = self.proj1(h).view(-1,self.hidden_dim)
        h_proj2 = self.proj2(h).permute(1,0)
        mm = torch.mm(h_proj1,h_proj2)
        mm = mm.view(N,self.hidden_dim,-1).permute(0,2,1) #[N, N, D]
        lr_e_global = mm[g.all_edges()[0],g.all_edges()[1],:] #[M,D]
        lr_e_global = self.edge_proj3(lr_e_global)

        e = lr_e_local + lr_e_global #baseline4        
        
        # bn=>relu=>dropout
        e = self.bn_node_lr_e(e)
        e = F.relu(e)
        e = F.dropout(e, 0.1, training=self.training)
        return e

class GatedGCNNet(nn.Module):
    
    def __init__(self, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        in_dim_edge = net_params['in_dim_edge']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.edge_feat = net_params['edge_feat']
        self.n_classes = n_classes
        self.device = net_params['device']
        
        self.layer_type = {
            "edgereprfeat": GatedGCNLayer,
            "edgefeat": GatedGCNLayerEdgeFeatOnly,
            "isotropic": GatedGCNLayerIsotropic,
        }.get(net_params['layer_type'], GatedGCNLayer)
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
#        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        self.layers = nn.ModuleList([ self.layer_type(hidden_dim, hidden_dim, dropout,
                                                      self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(self.layer_type(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(3*out_dim, n_classes)

        self.merg = MERG(in_dim, hidden_dim)
        

    def forward(self, g, h, e):
        lr_e = self.merg(g,h,e)
        e = lr_e

        h = self.embedding_h(h.float())
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h

        def _edge_feat(edges):
            e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            return {'e': e}
        g.apply_edges(_edge_feat)
        
        lr_e = torch.cat([g.edata['e'], e], dim=1)
        lr_e = self.MLP_layer(lr_e)
        #return g.edata['e']
        return lr_e
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss(weight=None)
        loss = criterion(pred, label)

        return loss
    