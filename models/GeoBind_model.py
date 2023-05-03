import torch
import os, sys
from .Segmentation import Conv
from torch import nn
import numpy as np
import time
from torch_scatter import scatter_max
def printf(args, *content):
    file = sys.stdout
    f_handler = open(os.path.join(args.checkpoints_dir, 'log.txt'), 'a+')
    sys.stdout = f_handler
    print(' '.join(content))
    f_handler.close()
    sys.stdout = file
    print(' '.join(content))

class EarlyStopping:
    def __init__(self, opt, patience_stop=10, patience_lr=5, verbose=False, delta=0.0001, path='check1.pth'):
        self.opt = opt
        self.stop_patience = patience_stop
        self.lr_patience = patience_lr
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_auc, model):
        if self.best_score is None:
            self.best_score = val_auc
            self.save_checkpoint(model)
            printf(self.opt, 'saving best model...')
        elif val_auc <= self.best_score + self.delta:
            self.counter += 1
            if self.counter == self.lr_patience:
                self.adjust_lr(model)
                # self.counter = 0

            if self.counter >= self.stop_patience:
                self.early_stop = True
        else:
            self.best_score = val_auc
            self.save_checkpoint(model)
            self.counter = 0
            printf(self.opt, 'saving best model...')

    def adjust_lr(self, model):
        lr = model.optimizer.param_groups[0]['lr']
        lr = lr/10
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = lr
        model.load_state_dict(torch.load(self.path))
        printf(self.opt, 'loading best model, changing learning rate to %.7f' % lr)

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

class GeoBind(nn.Module):
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation)
    --arch -> network type
    """
    def __init__(self, opt):
        super(GeoBind, self).__init__()
        self.opt = opt

        feature_type_dim = {'hmm':30, 'chemi':6, 'geo':1}
        in_channels = 0
        for item in self.opt.input_feature_type:
            in_channels += feature_type_dim[item]
        self.gpu_ids = opt.device
        self.device = torch.device('{}'.format(self.gpu_ids)) if self.gpu_ids else torch.device('cpu')

        I = in_channels
        E = opt.emb_dims
        H = opt.post_units

        self.lr = opt.lr
        self.nclasses = opt.nclasses

        # Segmentation network:
        self.conv=Conv(
            opt,
            in_channels=I,
            out_channels=E,
            n_layers=opt.n_layers,
            radius=opt.radius,
        ).to(self.device)

        self.net_out=nn.Sequential(
            nn.Linear(E, H),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(H, H),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(H, 2),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def set_input(self, data):
        data.to(self.device)
        self.P = {}
        feature = []
        for item in self.opt.input_feature_type:
            feature.append(data[item])
        if len(feature) == 0:
            raise ValueError("No input type matches GeoBind input feature type")
        self.P['f'] = torch.cat(feature, dim=1)
        self.P['xyz'] = data['xyz']
        self.P['nuv'] = data['nuv']
        self.P["input_features"] = self.P['f']
        self.P['batch'] = data['xyz_batch']
        self.P['y'] = data['y'].type(torch.long)
        self.P['site'] = data['site'].type(torch.long)
        self.P['site_point'] = data['site_point'].type(torch.long)
        pass

    def embed(self):
        torch.cuda.synchronize(device=self.device)
        torch.cuda.reset_max_memory_allocated(device=self.device)
        begin = time.time()
        self.conv.load_mesh(
            self.P["xyz"],
            triangles=None,
            nuv=self.P["nuv"],
            batch=self.P["batch"],
        )

        self.P["embedding"] = self.conv(self.P['input_features'])
        torch.cuda.synchronize(device=self.device)
        end = time.time()
        memory_usage = torch.cuda.max_memory_allocated(device=self.device)
        conv_time = end - begin
        return conv_time, memory_usage

    def optimize_parameters(self):
        self.train()
        conv_time, memory_usage = self.embed()
        self.P["iface_preds"] = self.net_out(self.P["embedding"]).squeeze()
        self.optimizer.zero_grad()
        self.loss = self.compute_loss()
        self.loss.backward()
        self.optimizer.step()
        if self.opt.loss_type == 'interface':
            return torch.softmax(self.P["iface_preds"], dim=1)[:, 1]
        else:
            iface_softmax = torch.softmax(self.P["iface_preds"], dim=1)[:, 1]
            return scatter_max(iface_softmax, self.P['site_point'], dim=0)[0]

    def test(self):
        """tests model
        returns the likelihood of sites or interfaces be binding
        """
        self.eval()
        with torch.no_grad():
            conv_time, memory_usage = self.embed()
            self.P["iface_preds"] = self.net_out(self.P["embedding"]).squeeze()
            self.loss = self.compute_loss()
        if self.opt.loss_type == 'interface':
            return torch.softmax(self.P["iface_preds"], dim=1)[:, 1]
        else:
            iface_softmax = torch.softmax(self.P["iface_preds"], dim=1)[:, 1]
            return scatter_max(iface_softmax, self.P['site_point'], dim=0)[0]

    def compute_loss(self):
        if self.opt.loss_type == 'interface':
            loss = self.criterion(self.P["iface_preds"], self.P['y'])
        else:
            _, pos = scatter_max(torch.softmax(self.P["iface_preds"], dim=1), self.P['site_point'], dim=0)
            pos = pos[:,1]
            self.preds_sites = torch.softmax(self.P["iface_preds"], dim=1)[pos]
            loss = self.criterion(torch.log(self.preds_sites), self.P['site'])
        return loss

    def load_network(self, which_epoch='best'):
        save_filename = '%s.pth' % which_epoch
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        self.load_state_dict(torch.load(save_path))
        printf(self.opt, 'best model loaded:', save_path)

    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % which_epoch
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        if torch.cuda.is_available():
            torch.save(self.cpu().state_dict(), save_path)
            self.cuda(self.device)
        else:
            torch.save(self.cpu().state_dict(), save_path)
