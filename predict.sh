python -W ignore predict.py \
--ligand RNA \
--checkpoints_dir /home/aoli/Documents/website/app/GeoBind/Pretrained_model/RNA \
--device cuda:0 \
--radius 12.0 \
--n_layers 4 \
--input_feature_type hmm chemi \
--batch_size 1 \
--lr 0.001 \
--loss_type interface