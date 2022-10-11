import argparse

parser = argparse.ArgumentParser(description="Network parameters")

# Main parameters
parser.add_argument(
    "--ligand", type=str, help="DNA or RNA", choices=['RNA', 'DNA', 'ATP', 'CA', 'MG', 'HEM', 'MN'], default='ATP'
)
parser.add_argument(
    "--checkpoints_dir",
    type=str,
    default="Pretrained_model/ATP",
    help="Where the log and model save",
)

parser.add_argument(
    "--input_feature_type",
    type=str,
    nargs='+',
    default=['hmm', 'chemi',], #attention the feature order should be the same with first time train when loaded.
    help="hmm:30, chemi:6, geo:1,",
)

parser.add_argument(
    "--loss_type",
    type=str,
    default='site', # interface or not
    help="loss type interface if sum of point cloud, others if sum of sites,",
)
parser.add_argument(
    "--use_mesh", type=bool, default=False, help="Use precomputed surfaces"
)

parser.add_argument("--profile", type=bool, default=False, help="Profile code")

parser.add_argument(
    "--n_layers", type=int, default=4, help="Number of convolutional layers"
)
parser.add_argument(
    "--radius", type=float, default=12.0,  help="Radius to use for the convolution"
)

# Training
parser.add_argument(
    "--batch_size", type=int, default=1, help="Number of proteins in a batch"
)

parser.add_argument(
    "--device", type=str, default="cuda:0", help="Which gpu/cpu to train on"
)

parser.add_argument("--seed", type=int, default=42, help="Random seed")

parser.add_argument(
    "--lr",
    type=float,
    default=0.001,
    help="learning rate",
)

parser.add_argument(
    "--nclasses",
    type=int,
    default=2,
    help="Where the log and model save",
)

parser.add_argument(
    "--emb_dims",
    type=int,
    default=64,
    help="Number of input features (+ 3 xyz coordinates for DGCNNs)",
)
parser.add_argument(
    "--post_units",
    type=int,
    default=64,
    help="Number of hidden units for the post-processing MLP",
)