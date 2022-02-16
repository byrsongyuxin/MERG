import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.gated_gcn_layer import GatedGCNLayer
from layers.mlp_readout_layer import MLPReadout

class GTP(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_thresh):
        super().__init__()
        in_dim = hidden_dim
        self.bn_node_lr_e = nn.BatchNorm1d(hidden_dim)
        
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
            
            h_single = self.embedding_h(g.ndata['feat'].to(h.device))
            h_proj1 = F.dropout(F.relu(self.bn_node_lr_g1(self.proj_g1(h_single))), 0.1, training=self.training).view(-1,self.hidden_dim)
            h_proj2 = F.dropout(F.relu(self.bn_node_lr_g2(self.proj_g2(h_single))), 0.1, training=self.training).permute(1,0)
            mm = torch.mm(h_proj1,h_proj2)
            mm = mm.view(N,self.hidden_dim,-1).permute(0,2,1) #[N, N, D]            
        
            mm = self.proj_g(mm).squeeze(-1)
            diag_mm = torch.diag(mm)
            diag_mm = torch.diag_embed(diag_mm)
            mm -= diag_mm
            
            matrix = F.softmax(mm, dim=0) * F.softmax(mm, dim=1)
    
            #binarized = BinarizedF()
            #matrix = binarized.apply(matrix) #(0/1)
            lr_connetion = torch.where(matrix>self.edge_thresh)
            g.add_edges(lr_connetion[0], lr_connetion[1])
            lr_gs.append(g)
        g = dgl.batch(lr_gs).to(h.device)

        return g

class MERG(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()

        in_dim = hidden_dim
        self.proj1 = nn.Linear(in_dim,hidden_dim**2) 
        self.proj2 = nn.Linear(in_dim,hidden_dim)
        self.src_embedding_h = nn.Embedding(in_dim_node, hidden_dim) 
        self.dst_embedding_h = nn.Embedding(in_dim_node, hidden_dim) 
        self.edge_proj = nn.Linear(2*in_dim,hidden_dim) 
        self.edge_proj3 = nn.Linear(hidden_dim,hidden_dim) 
        self.hidden_dim = hidden_dim 
        self.bn_node_lr_e = nn.BatchNorm1d(hidden_dim)    
        
    def forward(self, g, h, e):
        g.apply_edges(lambda edges: {'src' : edges.src['feat']})
        src = self.src_embedding_h(g.edata['src'].to(h.device)) #[M,D]
        g.apply_edges(lambda edges: {'dst' : edges.dst['feat']})
        dst = self.dst_embedding_h(g.edata['dst'].to(h.device)) #[M,D]
        edge = torch.cat((src,dst),-1) #[M,2]
        lr_e_local = self.edge_proj(edge) #[M,D]

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

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        in_dim_edge = 1 # edge_dim (feat is a float)
        hidden_dim = net_params['hidden_dim']
        n_classes = net_params['n_classes']
        dropout = net_params['dropout']
        n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = n_classes
        self.device = net_params['device']
        self.pos_enc = net_params['pos_enc']
        if self.pos_enc:
            pos_enc_dim = net_params['pos_enc_dim']
            self.embedding_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        
        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim) # node feat is an integer
        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim) # edge feat is a float
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                    self.batch_norm, self.residual) for _ in range(n_layers) ])
        self.MLP_layer = MLPReadout(hidden_dim, n_classes)        

        self.gtp = GTP(in_dim, hidden_dim, edge_thresh=0.5)
        self.merg = MERG(in_dim, hidden_dim)

    def forward(self, g, h, e, h_pos_enc=None):
        h = self.embedding_h(h)
        if self.pos_enc:
            h_pos_enc = self.embedding_pos_enc(h_pos_enc.float()) 
            h = h + h_pos_enc
        
        lr_g = self.gtp(g,h,e)
        g = lr_g
        lr_e = self.merg(g,h,e)
        e = lr_e
        
        # res gated convnets
        for conv in self.layers:
            h, e = conv(g, h, e)

        # output
        h_out = self.MLP_layer(h)

        return h_out
        

    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss

