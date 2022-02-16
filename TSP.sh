python main_TSP_edge_classification_best_model.py --dataset TSP \
--gpu_id 0 \
--config 'configs/TSP_edge_classification_GatedGCN_100k.json' --edge_feat True \
--batch_size 16 --out_dir ./output/TSP_GatedGCN/ \
--max_time 60 \


python main_TSP_edge_classification_best_model.py --dataset TSP \
--gpu_id 0 \
--config 'configs/TSP_edge_classification_GAT_edgereprfeat.json' --edge_feat True \
--batch_size 32 --out_dir ./output/TSP_GAT/ \
--max_time 60 \

