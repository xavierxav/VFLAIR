import sys, os
sys.path.append(os.pardir)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
import random
import time
import copy

# from models.vision import resnet18, MLP2
from utils.basic_functions import cross_entropy_for_onehot, multiclass_auc
from utils.communication_protocol_funcs import get_size_of

# from evaluates.attacks.attack_api import apply_attack
from utils.communication_protocol_funcs import compress_pred,Cache,ins_weight


tf.compat.v1.enable_eager_execution() 

STOPPING_ACC = {'mnist': 0.977, 'cifar10': 0.80, 'cifar100': 0.40,'diabetes':0.69,\
'nuswide': 0.88, 'breast_cancer_diagnose':0.88,'adult_income':0.84,'cora':0.72,\
'avazu':0.83,'criteo':0.74,'nursery':0.99,'credit':0.82}  # add more about stopping accuracy for different datasets when calculating the #communication-rounds needed


class MainTaskVFL(object):

    def __init__(self, args):
        self.args = args
        self.k = args.k
        self.device = args.device
        self.dataset_name = args.dataset
        # self.train_dataset = args.train_dst
        # self.val_dataset = args.test_dst
        # self.half_dim = args.half_dim
        self.epochs = args.main_epochs
        self.lr = args.main_lr
        self.batch_size = args.batch_size
        self.models_dict = args.model_list
        # self.num_classes = args.num_classes
        # self.num_class_list = args.num_class_list
        self.num_classes = args.num_classes
        self.exp_res_dir = args.exp_res_dir

        self.exp_res_path = args.exp_res_path
        self.parties = args.parties
        
        self.Q = args.Q # FedBCD

        self.parties_data = None
        self.gt_one_hot_label = None
        self.clean_one_hot_label  = None
        self.pred_list = []
        self.pred_list_clone = []
        self.pred_gradients_list = []
        self.pred_gradients_list_clone = []
        
        # FedBCD related
        self.local_pred_list = []
        self.local_pred_list_clone = []
        self.local_pred_gradients_list = []
        self.local_pred_gradients_list_clone = []
        
        self.loss = []
        self.train_acc = []
        self.test_acc = []
        self.test_auc = []
        self.flag = 1
        self.stopping_iter = 0
        self.stopping_time = 0.0
        self.stopping_commu_cost = 0
        self.communication_cost = 0


        # Early Stop
        self.early_stop_threshold = args.early_stop_threshold
        self.final_epoch = 0
        self.current_epoch = 0
        self.current_step = 0

        # some state of VFL throughout training process
        self.first_epoch_state = None
        self.middle_epoch_state = None
        self.final_state = None
        # self.final_epoch_state = None # <-- this is save in the above parameters

        self.num_update_per_batch = args.num_update_per_batch
        self.num_batch_per_workset = args.Q #args.num_batch_per_workset
        self.max_staleness = self.num_update_per_batch*self.num_batch_per_workset 
  
    def pred_transmit(self): # Active party gets pred from passive parties
        for ik in range(self.k):
            pred, pred_detach = self.parties[ik].give_pred()

            if ik == (self.k-1): # Active party update local pred
                pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                self.parties[self.k-1].receive_pred(pred_clone, ik) 
            
            if ik < (self.k-1): # Passive party sends pred for aggregation
                ########### communication_protocols ###########
                if self.args.communication_protocol in ['Quantization','Topk']:
                    pred_detach = compress_pred( self.args ,pred_detach , self.parties[ik].local_gradient,\
                                    self.current_epoch, self.current_step).to(self.args.device)
                ########### communication_protocols ###########
                pred_clone = torch.autograd.Variable(pred_detach, requires_grad=True).to(self.args.device)
                
                self.communication_cost += get_size_of(pred_clone) #MB
                
                self.parties[self.k-1].receive_pred(pred_clone, ik) 
    
    def gradient_transmit(self):  # Active party sends gradient to passive parties
        gradient = self.parties[self.k-1].give_gradient() # gradient_clone
        # gradient is a list of gradients for each party

        for _i in range(len(gradient)-1): #len(gradient)-1 because gradient of the active party is not sent
            self.communication_cost += get_size_of(gradient[_i+1])#MB
        
        # active party transfer gradient to passive parties and gets its own gradient
        for ik in range(self.k):
            self.parties[ik].local_gradient = gradient[ik]
        return
    
    def label_to_one_hot(self, target, num_classes=10):
        try:
            _ = target.size()[1]
            # print("use target itself", target.size())
            onehot_target = target.type(torch.float32).to(self.device)
        except:
            target = torch.unsqueeze(target, 1).to(self.device)
            # print("use unsqueezed target", target.size())
            onehot_target = torch.zeros(target.size(0), num_classes, device=self.device)
            onehot_target.scatter_(1, target, 1)
        return onehot_target

    def LR_Decay(self,i_epoch):
        for ik in range(self.k):
            self.parties[ik].LR_decay(i_epoch)
        self.parties[self.k-1].global_LR_decay(i_epoch)
        
    def train_batch(self, parties_data, batch_label):
        '''
        batch_label: self.gt_one_hot_label   may be noisy
        '''
        gt_one_hot_label = batch_label
        
        self.parties[self.k-1].gt_one_hot_label = gt_one_hot_label
        # allocate data to each party
        for ik in range(self.k):
            self.parties[ik].obtain_local_data(parties_data[ik][0])

        # ====== normal vertical federated learning ======
        torch.autograd.set_detect_anomaly(True)
        # ======== Commu ===========
        if self.args.communication_protocol in ['Vanilla','FedBCD_p','Quantization','Topk'] or self.Q ==1 : # parallel FedBCD & noBCD situation
            
            # exchange info between parties
            self.pred_transmit() 
            self.gradient_transmit()
            
            # update parameters for active party
            self.parties[self.k-1].global_backward()
            self.parties[self.k-1].local_backward()
            
            # update parameters for passive parties
            for ik in range(self.k - 1):
                self.parties[ik].local_backward()
                _pred, _pred_clone= self.parties[ik].give_pred()
                
                # if FedBCD_p, passive party gets global model and stale pred from other parties
                if self.Q > 1:
                    self.parties[ik].global_model = copy.deepcopy(self.parties[self.k-1].global_model)
                    
                    # passive parties get staled data for local updates
                    # Initialize an empty list to store the processed tensors
                    updated_pred_received = []
                    # Loop through each tensor in the original list
                    for tensor in self.parties[self.k-1].pred_received:
                        # Detach the tensor from its current computation graph
                        detached_tensor = tensor.detach()
                        
                        # Clone the detached tensor to create a new tensor
                        cloned_tensor = detached_tensor.clone()
                        
                        # Enable gradient tracking on the cloned tensor
                        cloned_tensor.requires_grad_(True)
                        
                        # Move the tensor to the device
                        cuda_tensor = cloned_tensor.to(self.args.device)
                        
                        # Append the processed tensor to the list
                        updated_pred_received.append(cuda_tensor)
                    # Assign the newly created list of tensors to the appropriate party
                    self.parties[ik].pred_received = updated_pred_received
                    
                    for param in self.parties[ik].global_model.parameters():
                        self.communication_cost += get_size_of(param) #MB
                    self.communication_cost += get_size_of(self.parties[self.k-1].pred_received[0]) * (self.k - 1) #each passive party gets stale pred from other parties

            for q in range(self.Q - 1): # FedBCD: additional iterations without info exchange 
                # for passive party, do local update without info exchange
                for ik in range(self.k-1):
                    _pred, _pred_clone= self.parties[ik].give_pred()
                    self.parties[ik].pred_received[ik] = _pred_clone.requires_grad_(True).to(self.args.device)
                    self.parties[ik].update_local_gradient_BCD(gt_one_hot_label)

                    self.parties[ik].local_backward()

            for q in range(self.Q - 1):
                # for active party, do local and global update without info exchange
                _pred, _pred_clone = self.parties[self.k-1].give_pred() 
                _gradient = self.parties[self.k-1].give_gradient()
                self.parties[self.k-1].global_backward()
                self.parties[self.k-1].local_backward()
            
        elif self.args.communication_protocol in ['CELU']:
            for q in range(self.Q):
                if (q == 0) or (batch_label.shape[0] != self.args.batch_size): 
                    # exchange info between parties
                    self.pred_transmit() 
                    self.gradient_transmit() 
                    # update parameters for all parties
                    for ik in range(self.k):
                        self.parties[ik].local_backward()
                    self.parties[self.k-1].global_backward()

                    if (batch_label.shape[0] == self.args.batch_size): # available batch to cache
                        for ik in range(self.k):
                            batch = self.num_total_comms # current batch id
                            self.parties[ik].cache.put(batch, self.parties[ik].local_pred,\
                                self.parties[ik].local_gradient, self.num_total_comms + self.parties[ik].num_local_updates)
                else: 
                    for ik in range(self.k):
                        # Sample from cache
                        batch, val = self.parties[ik].cache.sample(self.parties[ik].prev_batches)
                        batch_cached_pred, batch_cached_grad, \
                            batch_cached_at, batch_num_update \
                                = val
                        
                        _pred, _pred_detach = self.parties[ik].give_pred()
                        weight = ins_weight(_pred_detach,batch_cached_pred,self.args.smi_thresh) # ins weight
                        
                        # Using this batch for backward
                        if (ik == self.k-1): # active
                            self.parties[ik].update_local_gradient(batch_cached_grad)
                            self.parties[ik].local_backward(weight)
                            self.parties[ik].global_backward()
                        else:
                            self.parties[ik].receive_gradient(batch_cached_grad)
                            self.parties[ik].local_backward(weight)


                        # Mark used once for this batch + check staleness
                        self.parties[ik].cache.inc(batch)
                        if (self.num_total_comms + self.parties[ik].num_local_updates - batch_cached_at >= self.max_staleness) or\
                            (batch_num_update + 1 >= self.num_update_per_batch):
                            self.parties[ik].cache.remove(batch)
                        
            
                        self.parties[ik].prev_batches.append(batch)
                        self.parties[ik].prev_batches = self.parties[ik].prev_batches[1:]#[-(num_batch_per_workset - 1):]
                        self.parties[ik].num_local_updates += 1

        elif self.args.communication_protocol in ['FedBCD_s']: # Sequential FedBCD_s
            for q in range(self.Q):
                if q == 0: 
                    #first iteration, active party gets pred from passsive party
                    self.pred_transmit() 
                    _gradient = self.parties[self.k-1].give_gradient()
                    if len(_gradient)>1:
                        for _i in range(len(_gradient)-1):
                            self.communication_cost += get_size_of(_gradient[_i+1])#MB
                    # active party: update parameters 
                    self.parties[self.k-1].local_backward()
                    self.parties[self.k-1].global_backward()
                else: 
                    # active party do additional iterations without info exchange
                    self.parties[self.k-1].give_pred()
                    _gradient = self.parties[self.k-1].give_gradient()
                    self.parties[self.k-1].local_backward()
                    self.parties[self.k-1].global_backward()

            # active party transmit grad to passive parties
            self.gradient_transmit() 

            # passive party do Q iterations
            for _q in range(self.Q):
                for ik in range(self.k-1): 
                    _pred, _pred_clone= self.parties[ik].give_pred() 
                    self.parties[ik].local_backward() 
        else:
            assert 1>2 , 'Communication Protocol not provided'
        # ============= Commu ===================
        

        pred = self.parties[self.k-1].global_pred
        loss = self.parties[self.k-1].global_loss
        predict_prob = F.softmax(pred, dim=-1)

        suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(batch_label, dim=-1)).item()
        train_acc = suc_cnt / predict_prob.shape[0]
        
        return loss.item(), train_acc

    def train(self):

        print_every = 1

        for ik in range(self.k):
            self.parties[ik].prepare_data_loader(batch_size=self.batch_size)

        test_acc = 0.0
        # Early Stop
        last_loss = 1000000
        early_stop_count = 0
        LR_passive_list = []
        LR_active_list = []
        self.num_total_comms = 0
        total_time = 0.0
        flag = 0
        self.current_epoch = 0
        start_time = time.time()
        for i_epoch in range(self.epochs):
            self.current_epoch = i_epoch
            i = -1
            data_loader_list = [self.parties[ik].train_loader for ik in range(self.k)]

            self.current_step = 0
            for parties_data in zip(*data_loader_list):
                self.gt_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
                self.gt_one_hot_label = self.gt_one_hot_label.to(self.device)
                
                self.parties_data = parties_data

                i += 1
                for ik in range(self.k):
                    self.parties[ik].local_model.train()
                self.parties[self.k-1].global_model.train()
                
                enter_time = time.time()
                loss , train_acc = self.train_batch(self.parties_data,self.gt_one_hot_label)
                self.loss.append(loss)
                self.train_acc.append(train_acc)
                exit_time = time.time()
                total_time += (exit_time-enter_time)
                self.num_total_comms = self.num_total_comms + 1
                if self.num_total_comms % 10 == 0:
                    print(f"total time for {self.num_total_comms} communication is {total_time}")
                if self.train_acc[-1] > STOPPING_ACC[str(self.args.dataset)] and flag == 0:
                    self.stopping_time = total_time
                    self.stopping_iter = self.num_total_comms
                    self.stopping_commu_cost = self.communication_cost
                    flag = 1

                self.current_step = self.current_step + 1

            self.trained_models = self.save_state(True)
            if self.args.save_model == True:
                self.save_trained_models()

            # LR decay
            self.LR_Decay(i_epoch)
            # LR record
            if self.args.k == 2:
                LR_passive_list.append(self.parties[0].give_current_lr())
                LR_active_list.append(self.parties[1].give_current_lr())

            # validation
            if (i + 1) % print_every == 0:
                print("validate and test")
                for ik in range(self.k):
                    self.parties[ik].local_model.eval()
                self.parties[self.k-1].global_model.eval()
                
                suc_cnt = 0
                sample_cnt = 0
                test_preds = []
                test_targets = []
                with torch.no_grad():
                    data_loader_list = [self.parties[ik].test_loader for ik in range(self.k)]
                    for parties_data in zip(*data_loader_list):

                        gt_val_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes)
                        gt_val_one_hot_label = gt_val_one_hot_label.to(self.device)
        
                        pred_list = []
                        for ik in range(self.k):
                            _local_pred = self.parties[ik].local_model(parties_data[ik][0])
                            pred_list.append(_local_pred)

                        # Normal Evaluation
                        test_logit, test_loss = self.parties[self.k-1].aggregate(pred_list, gt_val_one_hot_label)
                        enc_predict_prob = F.softmax(test_logit, dim=-1)

                        test_preds.append(list(enc_predict_prob.detach().cpu().numpy()))
                        predict_label = torch.argmax(enc_predict_prob, dim=-1)

                        actual_label = torch.argmax(gt_val_one_hot_label, dim=-1)
                        sample_cnt += predict_label.shape[0]
                        suc_cnt += torch.sum(predict_label == actual_label).item()
                        test_targets.append(list(gt_val_one_hot_label.detach().cpu().numpy()))
                        
                    self.test_acc.append(suc_cnt / float(sample_cnt))
                    test_preds = np.vstack(test_preds)
                    test_targets = np.vstack(test_targets)
                    self.test_auc.append(np.mean(multiclass_auc(test_targets, test_preds)))

                    print('Epoch {:.2f}% \t train_loss:{:.4f} train_acc:{:.4f} test_acc:{:.4f} test_auc:{:.4f}'.format(
                        (i_epoch+1)/self.epochs*100, self.loss[-1], self.train_acc[-1], self.test_acc[-1], self.test_auc[-1]))
                    
                    self.final_epoch = i_epoch
        
        self.final_state = self.save_state(True) 
        self.final_state.update(self.save_state(False)) 
        self.final_state.update(self.save_party_data()) 
        

        return self.test_acc[-1],self.stopping_iter,self.stopping_time,self.stopping_commu_cost

    def save_state(self, BEFORE_MODEL_UPDATE=True):
        if BEFORE_MODEL_UPDATE:
            return {
                "model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)],
                "global_model":copy.deepcopy(self.parties[self.args.k-1].global_model),
                # type(model) = <class 'xxxx.ModelName'>
                "model_names": [str(type(self.parties[ik].local_model)).split('.')[-1].split('\'')[-2] for ik in range(self.args.k)]+[str(type(self.parties[self.args.k-1].global_model)).split('.')[-1].split('\'')[-2]]
            
            }
        else:
            return {
                # "model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)]+[self.parties[self.args.k-1].global_model],
                "data": copy.deepcopy(self.parties_data), 
                "label": copy.deepcopy(self.gt_one_hot_label),
                "predict": [copy.deepcopy(self.parties[ik].local_pred_clone) for ik in range(self.k)],
                "gradient": [copy.deepcopy(self.parties[ik].local_gradient) for ik in range(self.k)],
                "local_model_gradient": [copy.deepcopy(self.parties[ik].weights_grad_a) for ik in range(self.k)],
                "train_acc": copy.deepcopy(self.train_acc),
                "loss": copy.deepcopy(self.loss),
                "global_pred":self.parties[self.k-1].global_pred,
                "final_model": [copy.deepcopy(self.parties[ik].local_model) for ik in range(self.args.k)],
                "final_global_model":copy.deepcopy(self.parties[self.args.k-1].global_model),
                
            }

    def save_party_data(self):
        return {
            "aux_data": [copy.deepcopy(self.parties[ik].aux_data) for ik in range(self.k)],
            "train_data": [copy.deepcopy(self.parties[ik].train_data) for ik in range(self.k)],
            "test_data": [copy.deepcopy(self.parties[ik].test_data) for ik in range(self.k)],
            "aux_label": [copy.deepcopy(self.parties[ik].aux_label) for ik in range(self.k)],
            "train_label": [copy.deepcopy(self.parties[ik].train_label) for ik in range(self.k)],
            "test_label": [copy.deepcopy(self.parties[ik].test_label) for ik in range(self.k)],
            "aux_attribute": [copy.deepcopy(self.parties[ik].aux_attribute) for ik in range(self.k)],
            "train_attribute": [copy.deepcopy(self.parties[ik].train_attribute) for ik in range(self.k)],
            "test_attribute": [copy.deepcopy(self.parties[ik].test_attribute) for ik in range(self.k)],
            "aux_loader": [copy.deepcopy(self.parties[ik].aux_loader) for ik in range(self.k)],
            "train_loader": [copy.deepcopy(self.parties[ik].train_loader) for ik in range(self.k)],
            "test_loader": [copy.deepcopy(self.parties[ik].test_loader) for ik in range(self.k)],
            "batchsize": self.args.batch_size,
            "num_classes": self.args.num_classes
        }
               
    def save_trained_models(self):
        dir_path = self.exp_res_dir + f'trained_models/parties{self.k}_topmodel{self.args.apply_trainable_layer}_epoch{self.epochs}/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if self.args.apply_defense:
            file_path = dir_path + f'{self.args.defense_name}_{self.args.defense_configs}.pkl'
        else:
            file_path = dir_path + 'NoDefense.pkl'
        torch.save(([self.trained_models["model"][i].state_dict() for i in range(len(self.trained_models["model"]))],
                    self.trained_models["model_names"]), 
                  file_path)
