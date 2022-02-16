#python main_COLLAB_edge_classification_best_model.py --dataset OGBL-COLLAB \
#--gpu_id 0 --config 'configs/COLLAB_edge_classification_GatedGCN_40k.json' \
#--out_dir ./output/COLLAB_GatedGCN/ \
#--dropout 0.1 --max_time 60 \

#--config 'configs/COLLAB_edge_classification_GatedGCN_PE_40k.json'

python main_COLLAB_edge_classification_best_model.py --dataset OGBL-COLLAB \
--gpu_id 0 --config 'configs/COLLAB_edge_classification_GAT_edgereprfeat.json' \
--out_dir ./output/COLLAB_GAT/ \
--dropout 0.1 --max_time 60 \






