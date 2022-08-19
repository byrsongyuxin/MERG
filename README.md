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

