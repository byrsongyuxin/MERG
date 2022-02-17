import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import dgl
from timm.models.layers import trunc_normal_
from torch.autograd import Function
"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""
from layers.gated_gcn_layer import GatedGCNLayer
from layers.mlp_readout_layer import MLPReadout

'''
class BinarizedF(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        a = torch.ones_like(input)
        b = torch.zeros_like(input)
        output = torch.where(input>=0.5,a,b)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        input_abs = torch.abs(input)
        ones = torch.ones_like(input)
        zeros = torch.zeros_like(input)
        input_grad = torch.where(input_abs<=1,ones, zeros)
        return input_grad
#binarized = BinarizedF()
#matrix = binarized.apply(matrix)
'''

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
            #matrix = torch.sigmoid(mm)
            #matrix = F.softmax(mm, dim=0)
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
        self.bn_node_lr_e_local = nn.BatchNorm1d(hidden_dim)
        self.bn_node_lr_e_global = nn.BatchNorm1d(hidden_dim)
        self.proj1 = nn.Linear(in_dim,hidden_dim**2)
        self.proj2 = nn.Linear(in_dim,hidden_dim)
        self.edge_proj = nn.Conv1d(in_channels=2,out_channels=1,kernel_size=3,padding=1)
        self.edge_proj2 = nn.Linear(in_dim,hidden_dim)
        self.edge_proj3 = nn.Linear(hidden_dim,hidden_dim)
        self.hidden_dim = hidden_dim
        #self.bn_local = nn.BatchNorm1d(in_dim) #baseline4'
        self.bn_local = nn.LayerNorm(in_dim)
        self.bn_global = nn.BatchNorm1d(hidden_dim) #baseline4

    def forward(self, g, h, e):
        # modified baseline4
        g.apply_edges(lambda edges: {'src' : edges.src['feat']})
        src = g.edata['src'].unsqueeze(1) #[M,1,D]
        g.apply_edges(lambda edges: {'dst' : edges.dst['feat']})
        dst = g.edata['dst'].unsqueeze(1) #[M,1,D]
        edge = torch.cat((src,dst),1).to(h.device) #[M,2,D]
        
        edge = self.bn_local(edge)
        lr_e_local = self.edge_proj(edge).squeeze(1)#[M,D]
        lr_e_local = F.dropout(F.relu(lr_e_local), 0.1, training=self.training)
        lr_e_local = self.edge_proj2(lr_e_local)
        
        N = h.shape[0]
        h_proj1 = F.dropout(F.relu(self.proj1(h)), 0.1, training=self.training)
        h_proj1 = h_proj1.view(-1,self.hidden_dim)
        h_proj2 = F.dropout(F.relu(self.proj2(h)), 0.1, training=self.training)
        h_proj2 = h_proj2.permute(1,0)
        mm = torch.mm(h_proj1,h_proj2)
        mm = mm.view(N,self.hidden_dim,-1).permute(0,2,1) #[N, N, D]
        lr_e_global = mm[g.all_edges()[0],g.all_edges()[1],:] #[M,D]
        
        lr_e_global = self.edge_proj3(self.bn_global(lr_e_global))
        # bn=>relu=>dropout
        lr_e_global = self.bn_node_lr_e_global(lr_e_global)
        lr_e_global = F.relu(lr_e_global)
        lr_e_global = F.dropout(lr_e_global, 0.1, training=self.training)  

        lr_e_local = self.bn_node_lr_e_local(lr_e_local)
        lr_e_local = F.relu(lr_e_local)
        lr_e_local = F.dropout(lr_e_local, 0.1, training=self.training) 
        
        e = lr_e_local + lr_e_global #baseline4

        return e


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x

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
        self.device = net_params['device']
        
        self.embedding_h = nn.Linear(in_dim, hidden_dim)
#        self.embedding_e = nn.Linear(in_dim_edge, hidden_dim)
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim, hidden_dim, dropout,
                                                    self.batch_norm, self.residual) for _ in range(n_layers-1) ]) 
        self.layers.append(GatedGCNLayer(hidden_dim, out_dim, dropout, self.batch_norm, self.residual))
        self.MLP_layer = MLPReadout(out_dim, n_classes)

        self.gtp = GTP(in_dim, hidden_dim, edge_thresh=0.3)
        self.merg = MERG(in_dim, hidden_dim)
        
    def forward(self, g, h, e):
        '''
        g [B*M]
        h [B*N,D]
        e [B*M,1]
        '''
        
        lr_g = self.gtp(g,h,e)
        g = lr_g
        lr_e = self.merg(g,h,e)
        e = lr_e
    
        h = self.embedding_h(h)
        
        # convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
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
    
    def lr_edge_trarnsformer(self, h):
        h = h.view(-1,self.node_num,self.in_dim) #[B,N,D]
        h = self.act(self.edge_proj(h)) #[B,N,N]
        weights = self.edgeTrans(h)
        
        return weights
