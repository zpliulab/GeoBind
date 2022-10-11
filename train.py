import time, os
from dataloader.dataloader import DataLoader
from models.GeoBind_model import GeoBind
from models.GeoBind_model import EarlyStopping
import numpy as np
import sys
from Arguments import parser
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from default_config.dir_options import dir_opts

def printf(args, *content):
    file = sys.stdout
    f_handler = open(os.path.join(args.checkpoints_dir, 'log.txt'), 'a+')
    sys.stdout = f_handler
    print(' '.join(content))
    f_handler.close()
    sys.stdout = file
    print(' '.join(content))

def aupr(true, proba):
    precision, recall, thresholds = precision_recall_curve(true, proba)
    result = auc(recall, precision)
    return result

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
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    base_dir = 'Dataset/' + args.ligand + '/'
    dir_opts = dir_opts(base_dir)
    printf(args)
    setattr(args, 'dir_opts', dir_opts)
    setattr(args, 'dataset', 'train')
    dataset_train = DataLoader(args)
    printf(args, '#training meshes = %d' % len(dataset_train))

    setattr(args, 'dataset', 'val')
    dataset_val = DataLoader(args)
    printf(args, '#val meshes = %d' % len(dataset_val))

    setattr(args, 'dataset', 'test')
    dataset_test = DataLoader(args)
    printf(args, '#test meshes = %d' % len(dataset_test))

    model = GeoBind(args)
    early_stop = EarlyStopping(opt=args, path=os.path.join(args.checkpoints_dir, 'best.pth'))
    # model.load_network()
    # writer = Writer(args)
    total_steps = 0

    auc_best = 0
    time1 = time.time()
    for epoch in range(1000):
        # printf(epoch)
        epoch_start_time = time.time()
        epoch_iter = 0

        loss_train = []; proba_train = []; true_train = []
        for i, data in enumerate(dataset_train):
            loss_, proba_ = train_batch(data, model)
            loss_train.append(loss_.detach().cpu().numpy())
            if args.loss_type == 'interface':
                proba_train.append(proba_.detach().cpu().numpy())
                true_train.append(model.P['y'].detach().cpu().numpy())
            else:
                proba_train.append(proba_.detach().cpu().numpy())
                true_train.append(model.P['site'].detach().cpu().numpy())

        #evaluate on validation dataset
        loss_val = []; proba_val = []; true_val = []
        for i, data in enumerate(dataset_val):
            loss_, proba_ = val_batch(data, model)
            loss_val.append(loss_.detach().cpu().numpy())
            # val_loss_on_epoch.append(loss_.detach().cpu().numpy())
            if args.loss_type == 'interface':
                proba_val.append(proba_.detach().cpu().numpy())
                true_val.append(model.P['y'].detach().cpu().numpy())
            else:
                proba_val.append(proba_.detach().cpu().numpy())
                true_val.append(model.P['site'].detach().cpu().numpy())

        #evaluate on testing dataset
        loss_test = []; proba_test = []; true_test = []
        for i, data in enumerate(dataset_test):
            loss_, proba_ = val_batch(data, model)
            loss_test.append(loss_.detach().cpu().numpy())
            if args.loss_type == 'interface':
                proba_test.append(proba_.detach().cpu().numpy())
                true_test.append(model.P['y'].detach().cpu().numpy())
            else:
                proba_test.append(proba_.detach().cpu().numpy())
                true_test.append(model.P['site'].detach().cpu().numpy())
        loss_train = np.average(np.array(loss_train))
        proba_train = np.concatenate(proba_train)
        true_train = np.concatenate(true_train)
        auc_train = roc_auc_score(true_train, proba_train)

        loss_val = np.average(np.array(loss_val))
        proba_val = np.concatenate(proba_val)
        true_val = np.concatenate(true_val)
        auc_val = roc_auc_score(true_val, proba_val)

        loss_test = np.average(np.array(loss_test))
        proba_test = np.concatenate(proba_test)
        true_test = np.concatenate(true_test)
        auc_test = roc_auc_score(true_test, proba_test)

        out_str = '\rEpoch:{:0>2d}, train_loss:{:.3f}, train_auc:{:.3f}, val_loss:{:.3f}, val_auc:{:.3f}, test_loss:{:.3f}, test_auc:{:.3f}'\
            .format(epoch, loss_train, auc_train,loss_val, auc_val, loss_test, auc_test)
        printf(args, out_str)
        early_stop(auc_val, model)
        if early_stop.early_stop == True:
            break