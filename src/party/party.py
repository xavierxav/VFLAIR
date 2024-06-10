import os
import sys
import numpy as np
import random
sys.path.append(os.pardir)

import torch
from torch.utils.data import DataLoader

from load.LoadDataset import load_dataset_per_party
from load.LoadModels import load_basic_models

from utils.basic_functions import cross_entropy_for_onehot
from utils.communication_protocol_funcs import Cache

from sys import getsizeof

class Party(object):
    def __init__(self, args, index):
        self.name = "party#" + str(index + 1)
        self.index = index
        self.args = args
        # data for training and testing
        self.half_dim = -1
        self.train_data = None
        self.test_data = None
        self.train_label = None
        self.test_label = None
        self.train_attribute = None
        self.test_attribute = None
        self.train_dst = None
        self.test_dst = None
        self.train_loader = None
        self.test_loader = None
        self.attribute_loader = None
        self.attribute_iter = None
        self.local_batch_data = None

        # local model
        self.local_model = None
        self.local_model_optimizer = None
        # global_model
        self.global_model = None
        self.global_model_optimizer = None


        self.prepare_data()
        
        self.prepare_model(args, index)

        self.local_gradient = None
        self.local_pred = None
        self.local_pred_clone = None
        self.pred_received = []
        for _ in range(args.k):
            self.pred_received.append([])

        self.cache = Cache()
        self.prev_batches = []
        self.num_local_updates = 0

    def give_pred(self):
        self.local_pred = self.local_model(self.local_batch_data)

        self.local_pred_clone = self.local_pred.detach().clone()

        return self.local_pred, self.local_pred_clone

    def prepare_data(self):
        # prepare raw data for training
        (
            self.args,
            self.half_dim,
            train_dst,
            test_dst,
        ) = load_dataset_per_party(self.args, self.index)
        if len(train_dst) == 2:
            self.train_data, self.train_label = train_dst
            self.test_data, self.test_label = test_dst
        elif len(train_dst) == 3:
            self.train_data, self.train_label, self.train_attribute = train_dst
            self.test_data, self.test_label, self.test_attribute = test_dst

    def prepare_data_loader(self, batch_size):
        self.train_loader = DataLoader(self.train_dst, batch_size=batch_size) # , shuffle=True
        self.test_loader = DataLoader(self.test_dst, batch_size=batch_size) # , shuffle=True
        if self.train_attribute != None:
            self.attribute_loader = DataLoader(self.train_attribute, batch_size=batch_size)
            self.attribute_iter = iter(self.attribute_loader)

    def prepare_model(self, args, index):
        # prepare model and optimizer
        (
            self.local_model,
            self.local_model_optimizer,
            self.global_model,
            self.global_model_optimizer,
        ) = load_basic_models(args, index)
    
    def give_current_lr(self):
        return (self.local_model_optimizer.state_dict()['param_groups'][0]['lr'])

    def LR_decay(self,i_epoch):
        eta_0 = self.args.main_lr
        eta_t = eta_0/(np.sqrt(i_epoch+1))
        for param_group in self.local_model_optimizer.param_groups:
            param_group['lr'] = eta_t 
            
    def obtain_local_data(self, data):
        self.local_batch_data = data
    
    def receive_pred(self, pred, giver_index):
        self.pred_received[giver_index] = pred

    def local_backward(self,weight=None):
        self.num_local_updates += 1 # another update
        
        # update local model
        self.local_model_optimizer.zero_grad()

        torch.autograd.set_detect_anomaly(True)
        if weight != None: # CELU
            ins_batch_cached_grad = torch.mul(weight.unsqueeze(1),self.local_gradient)
            self.weights_grad_a = torch.autograd.grad(
                self.local_pred,
                self.local_model.parameters(),
                grad_outputs=ins_batch_cached_grad,
                retain_graph=True
            )
        else:
            self.weights_grad_a = torch.autograd.grad(
                self.local_pred,
                self.local_model.parameters(),
                grad_outputs=self.local_gradient,
                retain_graph=True
            )
        for w, g in zip(self.local_model.parameters(), self.weights_grad_a):
            if w.requires_grad:
                w.grad = g.detach()
        self.local_model_optimizer.step()
