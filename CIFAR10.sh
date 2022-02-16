#python main_superpixels_graph_classification_best_model.py --dataset CIFAR10 \
#--gpu_id 1 --config 'configs/superpixels_graph_classification_GAT_CIFAR10_100k.json' \
#--batch_size 64 --out_dir ./output/CIFAR10_GAT/ \
#--dropout 0.1 --max_time 60 \

python main_superpixels_graph_classification_best_model.py --dataset CIFAR10 \
--gpu_id 1 --config 'configs/superpixels_graph_classification_GatedGCN_CIFAR10_100k.json' \
--batch_size 32 --out_dir ./output/CIFAR10_GatedGCN/ \
--dropout 0.1 --max_time 60 \
