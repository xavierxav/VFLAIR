import sys, os
sys.path.append(os.pardir)
import torch
from party.party import Party
import torch.nn.functional as F

class ActiveParty(Party):
    def __init__(self, args, index, train_dataset, test_dataset):
        super().__init__(args, index, train_dataset, test_dataset)
        
        self.label = None
        
        self.global_pred = None
        self.global_loss = None
        self.criterion = F.cross_entropy

    def aggregate(self, pred_list, label):
        pred = self.global_model(pred_list)
        loss = self.criterion(pred, label)
        return pred, loss

    def gradient_calculation(self, pred_list, loss):
        pred_gradients_list_clone = []
        for ik in range(self.args.k):
            pred_gradients = torch.autograd.grad(loss, pred_list[ik], retain_graph=True, create_graph=True)
            pred_gradients_list_clone.append(pred_gradients[0].detach().clone())
        return pred_gradients_list_clone
    
    def give_gradient(self):
        pred_list = self.pred_received 

        assert self.label is not None, 'give gradient:self.label == None'

        self.global_pred, self.global_loss = self.aggregate(pred_list, self.label)
        
        pred_gradients_list_clone = []
        for ik in range(self.args.k):
            pred_gradients = torch.autograd.grad(self.global_loss, pred_list[ik], retain_graph=True, create_graph=True)
            pred_gradients_list_clone.append(pred_gradients[0].detach().clone())
        
        return pred_gradients_list_clone
                     
    def global_backward(self):

        if self.global_model_optimizer != None: 
            # active party with trainable global layer
            _gradients = torch.autograd.grad(self.global_loss, self.global_pred, retain_graph=True)
            _gradients_clone = _gradients[0].detach().clone()
            
            # update global model
            self.global_model_optimizer.zero_grad()
            # trainable layer parameters
            if self.args.global_model.apply_trainable_layer == True:
                # load grads into parameters
                weights_grad_a = torch.autograd.grad(self.global_pred, self.global_model.parameters(), grad_outputs=_gradients_clone, retain_graph=True)
                for w, g in zip(self.global_model.parameters(), weights_grad_a):
                    if w.requires_grad:
                        w.grad = g.detach()
            # non-trainabel layer: no need to update
            self.global_model_optimizer.step()
    