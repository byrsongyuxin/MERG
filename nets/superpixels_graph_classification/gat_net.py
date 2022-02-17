import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""
from layers.gat_layer import GATLayer, CustomGATLayerEdgeReprFeat
from layers.mlp_readout_layer import MLPReadout

class GTP(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_thresh):
        super().__init__()
        
        self.proj_g1 = nn.Linear(in_dim,hidden_dim**2) #lr_g
        self.bn_node_lr_g1 = nn.BatchNorm1d(hidden_dim**2)
        self.proj_g2 = nn.Linear(in_dim,hidden_dim) #lr_g
        self.bn_node_lr_g2 = nn.BatchNorm1d(hidden_dim)
        self.hidden_dim = hidden_dim #lr_g
        self.proj_g = nn.Linear(hidden_dim, 1)
        self.edge_thresh = edge_thresh

    def forward(self, g, h, e):

        lr_gs = []
        gs = dgl.unbatch(g)
        for g in gs:
            N = g.number_of_nodes()
            h_single = g.ndata['feat'].to(h.device)
            h_proj1 = F.dropout(F.relu(self.bn_node_lr_g1(self.proj_g1(h_single))), 0.1, training=self.training).view(-1,self.hidden_dim)
            h_proj2 = F.dropout(F.relu(self.bn_node_lr_g2(self.proj_g2(h_single))), 0.1, training=self.training).permute(1,0)
            mm = torch.mm(h_proj1,h_proj2)
            mm = mm.view(N,self.hidden_dim,-1).permute(0,2,1) #[N, N, D]

            mm = self.proj_g(mm).squeeze(-1)
            diag_mm = torch.diag(mm)
            diag_mm = torch.diag_embed(diag_mm)
            mm -= diag_mm
            
#            matrix = torch.sigmoid(mm)
#            matrix = F.softmax(mm, dim=0)
            matrix = F.softmax(mm, dim=0) * F.softmax(mm, dim=1)
            
            lr_connetion = torch.where(matrix>self.edge_thresh)
            g.add_edges(lr_connetion[0], lr_connetion[1])
            lr_gs.append(g)
        g = dgl.batch(lr_gs).to(h.device)
        return g

class MERG(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads):
        super().__init__()
        self.in_dim = in_dim
        self.proj1 = nn.Linear(in_dim,hidden_dim**2)
        self.proj2 = nn.Linear(in_dim,hidden_dim) 
        self.edge_proj = nn.Conv1d(in_channels=2,out_channels=1,kernel_size=3,padding=1)
        self.edge_proj2 = nn.Linear(in_dim,hidden_dim * num_heads)
        self.edge_proj3 = nn.Linear(hidden_dim,hidden_dim * num_heads)
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
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([CustomGATLayerEdgeReprFeat(hidden_dim * num_heads, hidden_dim, num_heads,
                                              dropout, self.batch_norm, self.residual) for _ in range(n_layers-1)])
        self.layers.append(GATLayer(hidden_dim * num_heads, out_dim, 1, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)
        
        self.gtp = GTP(in_dim, hidden_dim, edge_thresh=0.3)
        self.merg = MERG(in_dim, hidden_dim, num_heads)


    def forward(self, g, h, e):
        lr_g = self.gtp(g,h,e)
        g = lr_g
        lr_e = self.merg(g,h,e)
        e = lr_e

        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        for conv in self.layers:
            h = conv(g, h, e)
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