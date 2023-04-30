import time, os
from dataloader.dataloader import DataLoader

from models.GeoBind_model import GeoBind

import numpy as np
import sys
import pandas as pd
from Arguments import parser
from torch_scatter import scatter_max
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, precision_recall_curve,auc
from default_config.dir_options import dir_opts

def choose_cutoff(true, proba):
    max_mcc = 0
    thre_max_mcc = 0
    for item in range(0,1000,1):
        item=item/1000
        predict_binary=np.zeros_like(proba)
        predict_binary[np.where(proba > item)[0]]=1
        mcc = matthews_corrcoef(true, predict_binary)
        if mcc > max_mcc:
            max_mcc = mcc
            thre_max_mcc = item
    predict_binary = np.zeros_like(proba)
    predict_binary[np.where(proba>thre_max_mcc)[0]]=1
    return predict_binary, thre_max_mcc

# sys.stdout=f_handler
def train_batch(data, model):
    model.set_input(data)
    out = model.optimize_parameters()
    loss = model.loss
    return loss, out

def val_batch(data, model):

    model.set_input(data)
    out = model.test()
    loss = model.loss
    return loss, out

if __name__ == '__main__':
    args = parser.parse_args()
    # if not os.path.exists(args.checkpoints_dir):
    #     os.mkdir(args.checkpoints_dir)
    base_dir = 'Dataset/' + args.ligand + '/'
    dir_opts = dir_opts(base_dir)
    setattr(args, 'dir_opts', dir_opts)
    setattr(args, 'dataset', 'val')
    dataset_val = DataLoader(args)
    setattr(args, 'dataset', 'test')
    dataset_test = DataLoader(args)
    print('#test meshes = %d' % len(dataset_test))

    model = GeoBind(args)
    model.load_network()


    proba_val = []
    true_val = []
    for i, data in enumerate(dataset_val):
        print(data.pdb_id)
        loss_, proba_ = val_batch(data, model)
        # val_loss_on_epoch.append(loss_.detach().cpu().numpy())
        if args.loss_type == 'interface':
            proba_val.append(proba_.detach().cpu().numpy())
            true_val.append(model.P['y'].detach().cpu().numpy())
        else:
            proba_val.append(proba_.detach().cpu().numpy())
            true_val.append(model.P['site'].detach().cpu().numpy())
        # print(data.pdb_id, 'interface:',auc_tmp,'site:',auc_tmp2)

    proba_test = []
    true_test = []

    for i, data in enumerate(dataset_test):
        loss_, proba_ = val_batch(data, model)
        # val_loss_on_epoch.append(loss_.detach().cpu().numpy())
        if args.loss_type == 'interface':
            proba_test.append(proba_.detach().cpu().numpy())
            true_test.append(model.P['y'].detach().cpu().numpy())
        else:
            proba_test.append(proba_.detach().cpu().numpy())
            true_test.append(model.P['site'].detach().cpu().numpy())
        # all_auc.append([data.pdb_id, auc_tmp2])

    proba_val = np.concatenate(proba_val)
    true_val = np.concatenate(true_val)
    proba_test = np.concatenate(proba_test)
    true_test = np.concatenate(true_test)

    val_predict_site, thre = choose_cutoff(true_val, proba_val)

    auc_test = roc_auc_score(true_test, proba_test)
    precision, recall, _ = precision_recall_curve(true_test, proba_test)

    # df.to_csv('./compare_with_other_methods/GeoBind/GeoBind_RNA.csv', index=None, header=None)

    aupr_site = auc(recall, precision)

    test_predict_site = np.zeros_like(proba_test)
    test_predict_site[proba_test>thre] = 1
    mcc = matthews_corrcoef(true_test, test_predict_site)
    recall = recall_score(true_test, test_predict_site)
    precision = precision_score(true_test, test_predict_site)
    f1 = f1_score(true_test, test_predict_site)

    # val_loss_on_epoch.append(val_loss_on_batches)
    # val_auc_on_epoch.append(auc_val)
    out_str  = ''
    out_str += '\rsite:, thre:{:.3f}, rec:{:.3f}, pre:{:.3f}, F1:{:.3f}, mcc:{:.3f}, auc:{:.3f}, aupr:{:.3f}, positive site:{}, negative site:{}\n'\
        .format(thre, recall, precision, f1, mcc, auc_test, aupr_site, np.sum(true_test), true_test.shape[0]-np.sum(true_test))

    sys.stdout.write(out_str)
    sys.stdout.flush()

