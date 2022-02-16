# Learning Task-specific Topology and Multi-dimensional EdgeFeatures for Graphs

<br>
* The implementation of paper: Learning Task-specific Topology and Multi-dimensional EdgeFeatures for Graphs. 
* Project based on DGL 0.4.2. PyTorch 1.7

## 1. Requirement
Setup the environment follow this [link](https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/docs/01_benchmark_installation.md).

## 2. Data Preparing
Follow this [link](https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/docs/02_download_datasets.md). to download datasets(MNIST,CIFAR10,CLUSTER,PATTERN,COLLAB,TSP).

## 3. Reproducibility
To reproduct the main article results in Tabel 1, please refer to the scripts(CIDAR10.sh, MNIST.sh, CLUSTER.sh, PATTERN.sh, COLLAB.sh, TSP.sh)

## the project structure:
```
TTME/
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

