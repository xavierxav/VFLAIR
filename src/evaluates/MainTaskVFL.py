import os
import torch
import torch.nn.functional as F
import numpy as np
import time
import copy
from utils.basic_functions import multiclass_auc
from utils.communication_protocol_funcs import get_size_of, compress_pred, ins_weight

STOPPING_ACC = {'mnist': 0.977, 'cifar10': 0.80, 'cifar100': 0.40,'diabetes':0.69,\
'nuswide': 0.88, 'breast_cancer_diagnose':0.88,'adult_income':0.84,'cora':0.72,\
'avazu':0.83,'criteo':0.74,'nursery':0.99,'credit':0.82, 'satellite':10}
# add more about stopping accuracy for different datasets when calculating the #communication-rounds needed


class MainTaskVFL(object):

    def __init__(self, args):
        self.args = args
        self.k = args.k
        self.device = args.runtime.device
        self.dataset_name = args.dataset.dataset_name
        self.epochs = args.epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.models_dict = args.model_list
        self.num_classes = args.dataset.num_classes
        self.parties = args.parties
        self.Q = args.communication.iteration_per_aggregation # number of iterations for FedBCD        
        self.parties_data = None
        self.gt_one_hot_label = None
        self.loss = []
        self.train_acc = []
        self.test_acc = []
        self.test_auc = []
        self.communication_cost = 0
        # Early Stop
        self.early_stop_threshold = args.early_stop_threshold
        self.final_epoch = 0
        self.current_epoch = 0
        self.current_step = 0
        self.final_state = None
  
    def pred_transmit(self): # Active party gets pred from passive parties
        for ik in range(self.k):
            self.parties[ik].update_local_pred()

            if ik == (self.k-1): # Active party
                pred_clone = self.parties[self.k-1].local_pred.detach().clone().requires_grad_(True).to(self.device)
                self.parties[self.k-1].receive_pred(pred_clone, self.k-1)
            
            else : # Passive party sends pred for aggregation
                ########### communication_protocols ###########
                if self.args.communication.communication_protocol in ['Quantization','Topk']:
                    pred_detach = compress_pred( self.args ,pred_detach , self.parties[ik].local_gradient,\
                                    self.current_epoch, self.current_step)
                ########### communication_protocols ###########
                pred_clone = self.parties[ik].local_pred.detach().clone().requires_grad_(True).to(self.device)
                
                if self.args.communication.get_communication_size:
                    self.communication_cost += get_size_of(pred_clone) #MB
                
                self.parties[self.k-1].receive_pred(pred_clone, ik)
    
    def gradient_transmit(self):  # Active party sends gradient to passive parties
        gradient = self.parties[self.k-1].give_gradient() # gradient_clone
        # gradient is a list of gradients for each party
        if self.args.communication.get_communication_size:
            for _i in range(len(gradient)-1): #len(gradient)-1 because gradient of the active party is not sent
                self.communication_cost += get_size_of(gradient[_i+1])#MB
        
        # active party transfer gradient to passive parties and gets its own gradient
        for ik in range(self.k):
            self.parties[ik].local_gradient = gradient[ik]
    
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
        
    def train_batch(self):
        self.parties[self.k-1].gt_one_hot_label = self.gt_one_hot_label
        # allocate data to each party
        for ik in range(self.k):
            self.parties[ik].obtain_local_data(self.parties_data[ik][0])

        # ====== normal vertical federated learning ======
        if self.args.runtime.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        # ======== Commu ===========
        if self.args.communication.communication_protocol in ['Vanilla','FedBCD_p','Quantization','Topk'] or self.Q ==1 : # parallel FedBCD & noBCD situation
            
            # exchange info between parties
            self.pred_transmit() 
            self.gradient_transmit()
            
            # update parameters for active party
            self.parties[self.k-1].global_backward()
            self.parties[self.k-1].local_backward()
            
            # update parameters for passive parties
            for ik in range(self.k - 1):
                self.parties[ik].local_backward()
                self.parties[ik].update_local_pred()
                
                # if FedBCD_p, passive party gets global model and stale pred from other parties
                if self.Q > 1:
                    self.parties[ik].global_model = copy.deepcopy(self.parties[self.k-1].global_model)
                    
                    # passive parties get staled data for local updates
                    updated_pred_received = [tensor.detach().clone().requires_grad_(True).to(self.device) for tensor in self.parties[self.k-1].pred_received]
                    # Assign the newly created list of tensors to the appropriate party
                    self.parties[ik].pred_received = updated_pred_received
                    
                    if self.args.communication.get_communication_size:
                        for param in self.parties[ik].global_model.parameters():
                            self.communication_cost += get_size_of(param) #MB
                        self.communication_cost += get_size_of(self.parties[self.k-1].pred_received[0]) * (self.k - 1) #each passive party gets stale pred from other parties

            for q in range(self.Q - 1): # FedBCD: additional iterations without info exchange 
                # for passive party, do local update without info exchange
                for ik in range(self.k-1):
                    _pred, _pred_clone= self.parties[ik].update_local_pred()
                    self.parties[ik].pred_received[ik] = _pred_clone.requires_grad_(True).to(self.device)
                    self.parties[ik].update_local_gradient_BCD(self.gt_one_hot_label)

                    self.parties[ik].local_backward()

            for q in range(self.Q - 1):
                # for active party, do local and global update without info exchange
                _pred, _pred_clone = self.parties[self.k-1].update_local_pred() 
                _gradient = self.parties[self.k-1].give_gradient()
                self.parties[self.k-1].global_backward()
                self.parties[self.k-1].local_backward()
            
        elif self.args.communication.communication_protocol in ['CELU']:
            for q in range(self.Q):
                if (q == 0) or (self.gt_one_hot_label.shape[0] != self.args.batch_size): 
                    # exchange info between parties
                    self.pred_transmit() 
                    self.gradient_transmit() 
                    # update parameters for all parties
                    for ik in range(self.k):
                        self.parties[ik].local_backward()
                    self.parties[self.k-1].global_backward()

                    if (self.gt_one_hot_label.shape[0] == self.args.batch_size): # available batch to cache
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
                        
                        _pred, _pred_detach = self.parties[ik].update_local_pred()
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

        elif self.args.communication.communication_protocol in ['FedBCD_s']: # Sequential FedBCD_s
            for q in range(self.Q):
                if q == 0: 
                    #first iteration, active party gets pred from passsive party
                    self.pred_transmit() 
                    _gradient = self.parties[self.k-1].give_gradient()

                    if self.args.communication.get_communication_size:
                        if len(_gradient)>1:
                            for _i in range(len(_gradient)-1):
                                self.communication_cost += get_size_of(_gradient[_i+1])#MB
                    # active party: update parameters 
                    self.parties[self.k-1].local_backward()
                    self.parties[self.k-1].global_backward()
                else: 
                    # active party do additional iterations without info exchange
                    self.parties[self.k-1].update_local_pred()
                    _gradient = self.parties[self.k-1].give_gradient()
                    self.parties[self.k-1].local_backward()
                    self.parties[self.k-1].global_backward()

            # active party transmit grad to passive parties
            self.gradient_transmit() 

            # passive party do Q iterations
            for _q in range(self.Q):
                for ik in range(self.k-1): 
                    _pred, _pred_clone= self.parties[ik].update_local_pred() 
                    self.parties[ik].local_backward() 
        else:
            assert 1>2 , 'Communication Protocol not provided'
        # ============= Commu ===================
        

        pred = self.parties[self.k-1].global_pred
        loss = self.parties[self.k-1].global_loss
        predict_prob = F.softmax(pred, dim=-1)

        suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == torch.argmax(self.gt_one_hot_label, dim=-1)).item()
        train_acc = suc_cnt / predict_prob.shape[0]
        
        return loss.item(), suc_cnt , predict_prob.shape[0]

    def train(self):
        print_every = 1
        for ik in range(self.k):
            self.parties[ik].prepare_data_loader(batch_size=self.batch_size)
        self.num_total_comms = 0
        total_time = 0.0
        self.current_epoch = 0
        for i_epoch in range(self.epochs):
            self.current_epoch = i_epoch
            i = -1
            data_loader_list = [self.parties[ik].train_loader for ik in range(self.k)]

            self.current_step = 0
            suc_cnt = 0
            sample_cnt = 0
            loss = 0.0
            for parties_data in zip(*data_loader_list):
                parties_data = [(data[0].to(self.device), data[1].to(self.device)) for data in parties_data]
                self.gt_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes).to(self.device)
                self.parties_data = parties_data

                i += 1
                for ik in range(self.k):
                    self.parties[ik].local_model.train()
                self.parties[self.k-1].global_model.train()
                
                enter_time = time.time()
                loss_batch , suc_cnt_batch , sample_cnt_batch = self.train_batch()
                loss += loss_batch * sample_cnt_batch
                suc_cnt += suc_cnt_batch
                sample_cnt += sample_cnt_batch
                exit_time = time.time()
                total_time += (exit_time-enter_time)
                self.num_total_comms = self.num_total_comms + 1
                if self.num_total_comms % 10 == 0:
                    print(f"total time for {self.num_total_comms} communication is {total_time}")

                self.current_step += 1
            
            self.loss.append(loss / float(sample_cnt))
            self.train_acc.append(suc_cnt / float(sample_cnt))
            self.trained_models = self.save_state(True)
            if self.args.runtime.save_model == True:
                self.save_trained_models()

            # LR decay
            # self.LR_Decay(i_epoch)

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

                        gt_val_one_hot_label = self.label_to_one_hot(parties_data[self.k-1][1], self.num_classes).to(self.device)
        
                        pred_list = []
                        for ik in range(self.k):
                            _local_pred = self.parties[ik].local_model(parties_data[ik][0].to(self.device))
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
                "predict": [copy.deepcopy(self.parties[ik].local_pred.detach().clone()) for ik in range(self.k)],
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
            "train_loader": [copy.deepcopy(self.parties[ik].train_loader) for ik in range(self.k)],
            "test_loader": [copy.deepcopy(self.parties[ik].test_loader) for ik in range(self.k)],
            "batchsize": self.args.batch_size,
            "num_classes": self.args.dataset.num_classes
        }

    def save_trained_models(self):
        dir_path = self.exp_res_dir + f'trained_models/parties{self.k}_topmodel{self.args.global_model.apply_trainable_layer}_epoch{self.epochs}/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_path = dir_path + 'NoDefense.pkl'
        torch.save(([self.trained_models["model"][i].state_dict() for i in range(len(self.trained_models["model"]))],
                    self.trained_models["model_names"]), 
                  file_path)
