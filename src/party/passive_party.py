import sys, os
sys.path.append(os.pardir)
import torch
from torch.utils.data import DataLoader
from party.party import Party
from dataset.party_dataset import PassiveDataset
from dataset.party_dataset import ActiveDataset
from utils.basic_functions import cross_entropy_for_onehot , l2_reg


class PassiveParty(Party):
    def __init__(self, args, index):
        super().__init__(args, index)


    def prepare_data(self, args, index):
        super().prepare_data(args, index)
        self.train_dst = PassiveDataset(self.train_data)
        self.test_dst = PassiveDataset(self.test_data)
    
    def update_local_gradient_BCD(self, gt_one_hot_label):
        
        pred = self.global_model(self.pred_received)
        # loss = cross_entropy_for_onehot(pred, gt_one_hot_label) + l2_reg(self.global_model)
        loss = cross_entropy_for_onehot(pred, gt_one_hot_label)

        grad = torch.autograd.grad(loss, self.pred_received[self.index], retain_graph=True, create_graph=True)
        grad = grad[0].detach().clone()

        self.local_gradient = grad