# MERG: Multi-dimensional Edge Representation Generation Layer for Graph Neural Networks
![image](https://github.com/byrsongyuxin/MERG/blob/main/pipeline.png)
<br>
* The implementation of paper "MERG: Multi-dimensional Edge Representation Generation Layer for Graph Neural Networks". 
* Project based on DGL 0.4.2. PyTorch 1.7

## 1. Requirement
Setup the environment follow this [link](https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/docs/01_benchmark_installation.md).

## 2. Data Preparing
Follow this [link](https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/docs/02_download_datasets.md). to download datasets(MNIST,CIFAR10,CLUSTER,PATTERN,COLLAB,TSP).

## 3. Reproducibility
To reproduct the main article results in Tabel 1, please refer to the scripts(CIFAR10.sh, MNIST.sh, CLUSTER.sh, PATTERN.sh, COLLAB.sh, TSP.sh)
![image](https://github.com/byrsongyuxin/MERG/blob/main/sota.jpg)

## 4. Code
This code is based on the benchmarking-gnns codebase. The core code to implement the MERG Module is layers/gated_gcn_layer.py or layers/gat_layer.py. It is a plug-and-play module to generate multi-dimension edge feature where both of its corresponding node pair and the global contextual information are considered.
```python
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
        

    def forward(self, g, h, e, batch_labels):
        g.apply_edges(lambda edges: {'src' : edges.src['h']})
        src = g.edata['src'].unsqueeze(1) #[M,1,D]
        g.apply_edges(lambda edges: {'dst' : edges.dst['h']})
        dst = g.edata['dst'].unsqueeze(1) #[M,1,D]
        edge = torch.cat((src,dst),1).to(h.device) #[M,2,D]
        
        edge = self.bn_local(edge)
        lr_e_local = self.edge_proj(edge).squeeze(1)#[M,D]
        lr_e_local = F.dropout(F.relu(lr_e_local), 0.1, training=self.training)
        lr_e_local = self.edge_proj2(lr_e_local) #generated local edge feature
    
        
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
        lr_e_global = F.dropout(lr_e_global, 0.1, training=self.training) #generated global edge feature

        lr_e_local = self.bn_node_lr_e_local(lr_e_local)
        lr_e_local = F.relu(lr_e_local)
        lr_e_local = F.dropout(lr_e_local, 0.1, training=self.training) 
        
        e = lr_e_local + lr_e_global + e # with residual connection

        return e
```
## the project structure:
```
MERG/
  data/
    superpixels/
      CIFAR10.pkl
      MNIST.pkl
    SBMs/
      SBM_CLUSTER.pkl
      SBM_PATTERN.pkl			
    TSP/
      TSP.pkl
  dataset/
    ogbl_collab_dgl/
  nets/
  layers/
  output/
  train/
  CIFAR10.sh
  MNIST.sh
  CLUSTER.sh
  PATTERN.sh
  TSP.sh
  COLLAB.sh
```

## Acknowledgments
Our code is based on [benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns)
<br><br><br>

