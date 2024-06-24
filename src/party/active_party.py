import sys, os
sys.path.append(os.pardir)
import numpy as np
import torch
from party.party import Party
from utils.basic_functions import cross_entropy_for_onehot
from dataset.party_dataset import ActiveDataset , ActiveSatelliteDataset

class ActiveParty(Party):
    def __init__(self, args, index, train_dataset, test_dataset):
        super().__init__(args, index, train_dataset, test_dataset)
        
        self.gt_one_hot_label = None
        
        self.global_pred = None
        self.global_loss = None
        self.criterion = cross_entropy_for_onehot

    def aggregate(self, pred_list, gt_one_hot_label):
        pred = self.global_model(pred_list)
        loss = self.criterion(pred, gt_one_hot_label)
        return pred, loss

    def gradient_calculation(self, pred_list, loss):
        pred_gradients_list_clone = []
        for ik in range(self.args.k):
            pred_gradients = torch.autograd.grad(loss, pred_list[ik], retain_graph=True, create_graph=True)
            pred_gradients_list_clone.append(pred_gradients[0].detach().clone())
        return pred_gradients_list_clone
    
    def give_gradient(self):
        pred_list = self.pred_received 

        assert self.gt_one_hot_label is not None, 'give gradient:self.gt_one_hot_label == None'

        self.global_pred, self.global_loss = self.aggregate(pred_list, self.gt_one_hot_label)
        
        pred_gradients_list_clone = []
        for ik in range(self.args.k):
            pred_gradients = torch.autograd.grad(self.global_loss, pred_list[ik], retain_graph=True, create_graph=True)
            pred_gradients_list_clone.append(pred_gradients[0].detach().clone())
        
        return pred_gradients_list_clone

    def global_LR_decay(self,i_epoch):
        if self.global_model_optimizer != None: 
            eta_0 = self.args.lr
            eta_t = eta_0/(np.sqrt(i_epoch+1))
            for param_group in self.global_model_optimizer.param_groups:
                param_group['lr'] = eta_t
                     
    def global_backward(self):

        if self.global_model_optimizer != None: 
            # active party with trainable global layer
            _gradients = torch.autograd.grad(self.global_loss, self.global_pred, retain_graph=True)
            _gradients_clone = _gradients[0].detach().clone()
            
            # update global model
            self.global_model_optimizer.zero_grad()
            parameters = []          
            # trainable layer parameters
            if self.args.global_model.apply_trainable_layer == True:
                # load grads into parameters
                weights_grad_a = torch.autograd.grad(self.global_pred, self.global_model.parameters(), grad_outputs=_gradients_clone, retain_graph=True)
                for w, g in zip(self.global_model.parameters(), weights_grad_a):
                    if w.requires_grad:
                        w.grad = g.detach()
            # non-trainabel layer: no need to update
            self.global_model_optimizer.step()

    def calculate_gradient_each_class(self, global_pred, local_pred_list, test=False):
        # print(f"global_pred.shape={global_pred.size()}") # (batch_size, num_classes)
        self.gradient_each_class = [[] for _ in range(global_pred.size(1))]
        one_hot_label = torch.zeros(global_pred.size()).to(global_pred.device)
        for ic in range(global_pred.size(1)):
            one_hot_label *= 0.0
            one_hot_label[:,ic] += 1.0
            
            loss = self.criterion(global_pred, one_hot_label)
            for ik in range(self.args.k):
                self.gradient_each_class[ic].append(torch.autograd.grad(loss, local_pred_list[ik], retain_graph=True, create_graph=True))
        # end of calculate_gradient_each_class, return nothing
    