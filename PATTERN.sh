python main_SBMs_node_classification_best_model.py --dataset SBM_PATTERN \
--gpu_id 0 --config 'configs/SBMs_node_clustering_GAT_CLUSTER_500k.json' \
--batch_size 16 --out_dir ./output/PATTERN_MERG_GAT/ \
--max_time 60 \

python main_SBMs_node_classification_best_model.py --dataset SBM_PATTERN \
--gpu_id 0 --config 'configs/SBMs_node_clustering_GatedGCN_CLUSTER_PE_500k.json' \
--batch_size 16 --out_dir ./output/PATTERN_MERG_GatedGCN/ \
--max_time 60 \
