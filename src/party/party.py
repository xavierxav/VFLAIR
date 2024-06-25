import os
import sys
sys.path.append(os.pardir)

import torch
from torch.utils.data import DataLoader
from load.LoadModels import load_basic_models


class Party(object):
    def __init__(self, args, index, train_dataset, test_dataset, num_workers=0):
        self.index = index
        self.args = args
        # data for training and testing
        self.train_dst = train_dataset
        self.test_dst = test_dataset
        self.train_loader = DataLoader(self.train_dst, batch_size=args.batch_size, num_workers=num_workers, shuffle=True)
        self.test_loader = DataLoader(self.test_dst, batch_size=args.batch_size, num_workers=num_workers, shuffle=False)
        self.local_batch_data = None

        # local model
        self.local_model , self.local_model_optimizer, self.global_model, self.global_model_optimizer = load_basic_models(args, index)

        self.local_gradient = None
        self.local_pred = None
        self.pred_received = []
        for _ in range(args.k):
            self.pred_received.append([])
        self.prev_batches = []

    def update_local_pred(self):
        self.local_pred = self.local_model(self.local_batch_data)
            
    def obtain_local_data(self, data):
        self.local_batch_data = data
    
    def receive_pred(self, pred, giver_index):
        self.pred_received[giver_index] = pred

    def local_backward(self):
        
        # update local model
        self.local_model_optimizer.zero_grad()
        if self.args.runtime.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        
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
