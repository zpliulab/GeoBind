python -W ignore train.py \
--ligand ATP \
--checkpoints_dir Dataset/result/ATP \
--device cuda:1 \
--radius 12.0 \
--n_layers 4 \
--input_feature_type hmm chemi \
--batch_size 1 \
--lr 0.001 \
--loss_type site