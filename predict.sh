python -W ignore predict.py \
--ligand CA \
--checkpoints_dir Dataset/result/CA \
--device cuda:0 \
--radius 12.0 \
--n_layers 4 \
--input_feature_type hmm chemi \
--batch_size 1 \
--lr 0.001 \
--loss_type site