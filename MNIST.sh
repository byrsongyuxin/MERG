python main_superpixels_graph_classification_best_model.py --dataset MNIST \
--gpu_id 0 --config 'configs/superpixels_graph_classification_GAT_MNIST_100k.json' \
--batch_size 128 --out_dir ./output/MNIST_MERG_GAT/ \
--dropout 0.1 --max_time 60 \

python main_superpixels_graph_classification_best_model.py --dataset MNIST \
--gpu_id 0 --config 'configs/superpixels_graph_classification_GatedGCN_MNIST_100k.json' \
--batch_size 64 --out_dir ./output/MNIST_MERG_Gated-GCN/ \
--dropout 0.1 --max_time 60 \
