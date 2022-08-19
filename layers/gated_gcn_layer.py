import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

import os
import numpy as np
import matplotlib.pyplot as plt

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""

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
        
        self.v_id = 0

    def forward(self, g, h, e, batch_labels):
        # modified baseline4
        g.apply_edges(lambda edges: {'src' : edges.src['h']})
        src = g.edata['src'].unsqueeze(1) #[M,1,D]
        g.apply_edges(lambda edges: {'dst' : edges.dst['h']})
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
        
        ######## visualize
        background = torch.zeros([N, N])
        background[g.all_edges()[0],g.all_edges()[1]] = 1
        print(background)
        
        '''
        #print('batch_labels: ', batch_labels.shape, batch_labels.dtype)
        label = torch.ones([N,N], dtype=torch.int64)*2
        label = label.cuda()

        label[g.all_edges()[0],g.all_edges()[1]] = batch_labels
        print('label', label.max(), label.min(), label.shape)
        '''
        
#        plt.imshow(background, cmap=plt.cm.inferno, vmin=0, vmax=1) # magma inferno
#        plt.colorbar()
#        plt.savefig(os.path.join('CIFAR10_visual','back'+str(self.v_id)+'.jpg'))
#        plt.close()
#        print('save, background: ', os.path.join('CIFAR10_visual','back'+str(self.v_id)+'.jpg'))
        
        
        if self.v_id < 100:
            vmm = mm.mean(-1)
            
            #vmm = F.softmax(vmm, dim=0) * F.softmax(vmm, dim=1)
            
            vmm = vmm.cpu()
            vmm = (vmm - vmm.min().item()) / (vmm.max().item() - vmm.min().item())
           
            vmm_np = vmm.numpy()
            
            print('vmm ', vmm, vmm.min(), vmm.max(), vmm.mean())
            plt.imshow(vmm_np, cmap=plt.cm.inferno, vmin=0, vmax=1) # magma inferno
            plt.gca().set_title("TSP",fontsize=20)
            plt.xlabel("Node Index",fontsize=20)
            plt.ylabel("Node Index",fontsize=20)
            plt.colorbar()
            plt.savefig(os.path.join('TSP_visual',str(self.v_id)+'.jpg'))
            print('save: ', os.path.join('TSP_visual',str(self.v_id)+'.jpg'))
            
            self.v_id += 1
            plt.close()
            #plt.show()
            
        else:
            import sys; sys.exit(0)
        '''
        
        #label_np = batch_labels.cpu().numpy()
        label_np = label.reshape([-1]).cpu().numpy()
        
        #mm = mm[g.all_edges()[0],g.all_edges()[1],:] ###!!!
        
        vmm_np = mm.reshape([-1, mm.shape[-1]]).cpu().numpy()
        np.save('label_np',label_np)
        np.save('vmm_np',vmm_np)
        
        print('vmm_np, ', vmm_np.shape)
        print('label_np ', label_np.shape)
        '''
        ########
        
        
        lr_e_global = mm[g.all_edges()[0],g.all_edges()[1],:] #[M,D]
        
        lr_e_global = self.edge_proj3(self.bn_global(lr_e_global))
        # bn=>relu=>dropout
        lr_e_global = self.bn_node_lr_e_global(lr_e_global)
        lr_e_global = F.relu(lr_e_global)
        lr_e_global = F.dropout(lr_e_global, 0.1, training=self.training)  

        lr_e_local = self.bn_node_lr_e_local(lr_e_local)
        lr_e_local = F.relu(lr_e_local)
        lr_e_local = F.dropout(lr_e_local, 0.1, training=self.training) 
        
        e = lr_e_local + lr_e_global + e #baseline4

        return e
    
class GatedGCNLayer(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False, edge_lr = False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)
        
        self.edge_lr = edge_lr
        if self.edge_lr:
            self.merg = MERG(input_dim, output_dim)
    
    def forward(self, g, h, e, batch_labels = None):
        
        
        h_in = h # for residual connection
        e_in = e # for residual connection
        
        g.ndata['h']  = h
        
        if self.edge_lr:
            e = self.merg(g, h, e, batch_labels)
            
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        g.edata['e']  = e 
        g.edata['Ce'] = self.C(e) 

        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'DEh'))
        g.edata['e'] = g.edata['DEh'] + g.edata['Ce']
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)
        #g.update_all(self.message_func,self.reduce_func) 
        h = g.ndata['h'] # result of graph convolution
        e = g.edata['e'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization  
            e = self.bn_node_e(e) # batch normalization  
        
        h = F.relu(h) # non-linear activation
        e = F.relu(e) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
            e = e_in + e # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)

    
##############################################################
#
# Additional layers for edge feature/representation analysis
#
##############################################################


class GatedGCNLayerEdgeFeatOnly(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)

    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h) 
        g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h) 
        #g.update_all(self.message_func,self.reduce_func) 
        g.apply_edges(fn.u_add_v('Dh', 'Eh', 'e'))
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-6)
        h = g.ndata['h'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization    
        
        h = F.relu(h) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)


##############################################################


class GatedGCNLayerIsotropic(nn.Module):
    """
        Param: []
    """
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(output_dim)

    
    def forward(self, g, h, e):
        
        h_in = h # for residual connection
        
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h) 
        g.ndata['Bh'] = self.B(h)
        #g.update_all(self.message_func,self.reduce_func) 
        g.update_all(fn.copy_u('Bh', 'm'), fn.sum('m', 'sum_h'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_h']
        h = g.ndata['h'] # result of graph convolution
        
        if self.batch_norm:
            h = self.bn_node_h(h) # batch normalization    
        
        h = F.relu(h) # non-linear activation
        
        if self.residual:
            h = h_in + h # residual connection
        
        h = F.dropout(h, self.dropout, training=self.training)
        
        return h, e
    
    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels)
    
