import torch.utils.data
import random, os, tqdm
import numpy as np
import _pickle as cPickle
from torch_geometric.data import Data
import torch_geometric

def collate(batch):
    """Creates mini-batch tensors
    We should build custom collate_fn rather than using default collate_fn
    """
    meta = {}
    keys = batch[0].keys()
    for key in keys:
        aa = [d[key] for d in batch]
        # bb = Batch.from_data_list(aa)
        meta.update({key: aa})
        # meta.update({key: [d[key] for d in batch]})
    return meta

class DataLoader:
    """multi-threaded data loading"""

    def __init__(self, opt):
        self.opt = opt
        self.dataset = self.CreateDataset()
        self.dataloader = torch_geometric.loader.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=True if opt.dataset == 'train' else False,
            follow_batch=["xyz", 'site'],
            collate_fn=collate)

    def CreateDataset(self):
        if self.opt.ligand == 'RNA':
            train_file = './Dataset_lists/GeoBind/RNA-663_Train.txt'
            test_file = './Dataset_lists/GeoBind/RNA-157_Test.txt'
            # train_file = './Dataset_lists/GraphBind/RNA-495_Train.txt'
            # test_file = './Dataset_lists//GraphBind/RNA-117_Test.txt'
        elif self.opt.ligand == 'DNA':
            train_file = './Dataset_lists/GeoBind/DNA-719_Train.txt'
            test_file = './Dataset_lists/GeoBind/DNA-179_Test.txt'
            # train_file = './Dataset_lists//GraphBind/DNA-573_Train.txt'
            # test_file = './Dataset_lists//GraphBind/DNA-129_Test.txt'
        elif self.opt.ligand == 'ATP':
            train_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/ATP-388_Train.txt'
            test_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/ATP-41_Test.txt'
        elif self.opt.ligand == 'CA':
            train_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/CA-1022_Train.txt'
            test_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/CA-515_Test.txt'
        elif self.opt.ligand == 'MG':
            train_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/MG-1194_Train.txt'
            test_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/MG-651_Test.txt'
        elif self.opt.ligand == 'HEM':
            train_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/HEM-175_Train.txt'
            test_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/HEM-96_Test.txt'
        elif self.opt.ligand == 'MN':
            train_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/MN-440_Train.txt'
            test_file = 'Dataset_lists/Ligands_by_DELIA_ATPbind/MN-144_Test.txt'


        train_val_list = []
        test_list = []
        with open(train_file, 'r') as pid:
            for line in pid.readlines():
                train_val_list.append(line.strip().split('\t')[0])
        with open(test_file, 'r') as pid:
            for line in pid.readlines():
                test_list.append(line.strip().split('\t')[0])
        random.seed(1995)
        random.shuffle(train_val_list)
        train_list = train_val_list[int(len(train_val_list) * 0.2):]
        valid_list = train_val_list[:int(len(train_val_list) * 0.2)]

        if self.opt.dataset == 'train':
            rbps = train_list
        elif self.opt.dataset == 'val':
            rbps = valid_list
        else:
            rbps = test_list

        meta =[]
        label = []
        pos_label = []
        success = []
        data_label_dir = self.opt.dir_opts['data_label']
        for rbp in tqdm.tqdm(rbps):
            # if rbp != '4pbu:O:O1':
            #     continue
            if os.path.exists(os.path.join(data_label_dir, rbp)):
                with open(os.path.join(data_label_dir, rbp), 'rb') as pid:
                    data = cPickle.load(pid)
                    data1 = Data(
                                    xyz=torch.from_numpy(data['xyz']).type(torch.float32),
                                    nuv=torch.from_numpy(data['nuv']).type(torch.float32).contiguous(),
                                    chemi=torch.zeros([data['xyz_type'].shape[0], 6]).scatter_(1,  torch.from_numpy(data['xyz_type']).type(torch.int64), 1),
                                    hmm=torch.from_numpy(data['hmm']).type(torch.float32),
                                    geo=torch.from_numpy(data['geo']).type(torch.float32),
                                    y=torch.tensor(data['y']),
                                    site=torch.tensor(data['site']).type(torch.float32),
                                    site_point=torch.tensor(data['site_point']),
                                    pdb_id=rbp,
                                )

                    label.append(data['site'])
                    this_label_num = np.sum(data['site'])
                    if this_label_num == 0:
                        continue
                    pos_label.append(this_label_num)
                    meta.append(data1)
                    success.append(rbp+'\n')
                    # break
            else:
                print(rbp)
        label = np.hstack(label)
        print(self.opt.dataset, len(meta), int(np.sum(label)), label.shape[0]-np.sum(label), np.sum(label)/(label.shape[0]-np.sum(label)))
        return meta

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= len(self.dataset):
                break
            yield data
