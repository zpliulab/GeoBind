import os
from GeoBindProcessor import GeoBindProcessor
from default_config.dir_options import dir_opts
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Network parameters")

    # Main parameters
    parser.add_argument(
        "--ligand", type=str, help="specific one ligand from [DNA, RNA, ATP, CA, HEM, MN, MG]", choices=['RNA', 'DNA', 'ATP', 'CA', 'HEM', 'MN','MG'], required=True
    )
    parser.add_argument(
        "--pdbid",
        type=str,
        default="",
        help="Specify a PDB to process. Or the program will process the PDBs in defaut list one by one",
    )
    parser.add_argument(
        "--Gaussian_window",
        type=float,
        default="12.0",
        help="Gaussian window Radius to use for the convolution",
    )
    return parser
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.ligand == 'RNA':
        train_file = './Dataset_lists/GeoBind/RNA-663_Train.txt'
        test_file = './Dataset_lists/GeoBind/RNA-157_Test.txt'
        # train_file = './Dataset_lists/GraphBind/RNA-495_Train.txt'
        # test_file = './Dataset_lists/GraphBind/RNA-117_Test.txt'
    if args.ligand == 'DNA':
        train_file = './Dataset_lists/GeoBind/DNA-719_Train.txt'
        test_file = './Dataset_lists/GeoBind/DNA-179_Test.txt'
        # train_file = './Dataset_lists/GraphBind/DNA-573_Train.txt'
        # test_file = './Dataset_lists/GraphBind/DNA-129_Test.txt'
    if args.ligand == 'ATP':
        train_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/ATP-388_Train.txt'
        test_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/ATP-41_Test.txt'
    if args.ligand == 'CA':
        train_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/CA-1022_Train.txt'
        test_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/CA-515_Test.txt'
    if args.ligand == 'HEM':
        train_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/HEM-175_Train.txt'
        test_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/HEM-96_Test.txt'
    if args.ligand == 'MN':
        train_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/MN-440_Train.txt'
        test_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/MN-144_Test.txt'
    if args.ligand == 'MG':
        train_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/MG-1194_Train.txt'
        test_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/MG-651_Test.txt'

    if not os.path.exists('Dataset/'):
        os.mkdir('Dataset/')
    base_dir = 'Dataset/'+ args.ligand+'/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dir_opts = dir_opts(base_dir)

    protein_list = []
    with open(train_file, 'r') as pid:
        for line in pid.readlines():
            protein_list.append(line.strip())

    with open(test_file, 'r') as pid:
        for line in pid.readlines():
            protein_list.append(line.strip())
    print(len(protein_list))
    if not os.path.exists(dir_opts['data_label']):
        os.mkdir(dir_opts['data_label'])
    for index, item in enumerate(protein_list):
        pair = item.split('\t')[0]
        anno = item.split('\t')[1] if len(item.split('\t'))>1 else None
        if args.pdbid != '' and args.pdbid!= pair:
            continue
        #if not specify the binding sites given in residue id. GeoBind will automatically compute the Binding sites
        if os.path.exists(os.path.join(dir_opts['data_label'], pair)):
            continue
        print(index, item)
        try:
            rbp=GeoBindProcessor(pair, anno, args.ligand, dir_opts, Gaussian_window=args.Gaussian_window)
            rbp.get_data()
        except:
            continue