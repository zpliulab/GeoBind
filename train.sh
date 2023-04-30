python -W ignore train.py \
--ligand RNA \
--checkpoints_dir Dataset/result/RNA2 \
--device cuda:1 \
--radius 12.0 \
--n_layers 4 \
--input_feature_type hmm chemi geo \
--batch_size 1 \
--lr 0.001 \
--loss_type interface