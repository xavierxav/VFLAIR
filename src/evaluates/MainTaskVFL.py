import torch
import torch.nn.functional as F
import time
import copy
from src.utils.basic_functions import get_size_of
from torchmetrics.classification import MulticlassAUROC


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
        self.label = None
        self.loss = []
        self.train_acc = []
        self.test_acc = []
        self.test_auc = []
        self.communication_cost = 0
  
    def pred_transmit(self): # Active party gets pred from passive parties
        for ik in range(self.k):
            self.parties[ik].update_local_pred()

            if ik == (self.k-1): # Active party
                pred_clone = self.parties[self.k-1].local_pred.detach().clone().requires_grad_(True).to(self.device)
                self.parties[self.k-1].receive_pred(pred_clone, self.k-1)
            
            else :
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

    def train_batch(self):
        self.parties[self.k-1].label = self.label
        # allocate data to each party
        for ik in range(self.k):
            self.parties[ik].obtain_local_data(self.parties_data[ik][0])

        # ====== normal vertical federated learning ======
        if self.args.runtime.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        # ======== Commu ===========
        if self.args.communication.communication_protocol in ['Vanilla','FedBCD_p'] or self.Q ==1 : # parallel FedBCD & noBCD situation
            
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
                    self.parties[ik].update_local_gradient_BCD(self.label)

                    self.parties[ik].local_backward()

            for q in range(self.Q - 1):
                # for active party, do local and global update without info exchange
                _pred, _pred_clone = self.parties[self.k-1].update_local_pred() 
                _gradient = self.parties[self.k-1].give_gradient()
                self.parties[self.k-1].global_backward()
                self.parties[self.k-1].local_backward()
            
        else:
            assert 1>2 , 'Communication Protocol not provided'
        # ============= Commu ===================
        

        pred = self.parties[self.k-1].global_pred
        loss = self.parties[self.k-1].global_loss
        predict_prob = F.softmax(pred, dim=-1)

        suc_cnt = torch.sum(torch.argmax(predict_prob, dim=-1) == self.label).item()
        train_acc = suc_cnt / predict_prob.shape[0]
        
        return loss.item(), suc_cnt , predict_prob.shape[0]

    def train(self):
        print_every = 1
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
                self.label = parties_data[self.k-1][1]
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

                        val_label = parties_data[self.k-1][1].to(self.device)
        
                        pred_list = []
                        for ik in range(self.k):
                            _local_pred = self.parties[ik].local_model(parties_data[ik][0].to(self.device))
                            pred_list.append(_local_pred)

                        # Normal Evaluation
                        test_logit, test_loss = self.parties[self.k-1].aggregate(pred_list, val_label)
                        enc_predict_prob = F.softmax(test_logit, dim=-1)

                        test_preds.append(enc_predict_prob.detach().cpu())
                        predict_label = torch.argmax(enc_predict_prob, dim=-1)

                        sample_cnt += predict_label.shape[0]
                        suc_cnt += torch.sum(predict_label == val_label).item()
                        test_targets.append(val_label.detach().cpu())
                        
                    self.test_acc.append(suc_cnt / float(sample_cnt))
                    test_preds = torch.cat(test_preds, dim=0)
                    test_targets = torch.cat(test_targets, dim=0)
                    auroc = MulticlassAUROC(num_classes=self.num_classes)
                    self.test_auc.append(auroc(test_preds, test_targets).item())

                    print('Epoch {:.2f}% \t train_loss:{:.4f} train_acc:{:.4f} test_acc:{:.4f} test_auc:{:.4f}'.format(
                        (i_epoch+1)/self.epochs*100, self.loss[-1], self.train_acc[-1], self.test_acc[-1], self.test_auc[-1]))