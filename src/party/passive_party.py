import sys, os
sys.path.append(os.pardir)
import torch
from party.party import Party
import torch.nn.functional as F


class PassiveParty(Party):
    def __init__(self, args, index, train_dataset, test_dataset):
        super().__init__(args, index, train_dataset, test_dataset)
    
    def update_local_gradient_BCD(self, label):
        
        pred = self.global_model(self.pred_received)
        # loss = F.cross_entropy(pred, label) + l2_reg(self.global_model)
        loss = F.cross_entropy(pred, label)

        grad = torch.autograd.grad(loss, self.pred_received[self.index], retain_graph=True, create_graph=True)
        grad = grad[0].detach().clone()

        self.local_gradient = grad