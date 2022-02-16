python main_SBMs_node_classification_best_model.py --dataset SBM_CLUSTER \
--gpu_id 0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_500k.json' \
--batch_size 32 --out_dir ./output/CLUSTER_GAT/ \
--dropout 0.1 --max_time 60 \

python main_SBMs_node_classification_best_model.py --dataset SBM_CLUSTER \
--gpu_id 0 --config 'configs/SBMs_node_clustering_GatedGCN_PATTERN_PE_500k.json' \
--batch_size 32 --out_dir ./output/CLUSTER_GatedGCN/ \
--dropout 0.1 --max_time 60 \