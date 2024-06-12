import os
import numpy as np
import pandas as pd

import random
import argparse
import torch

from load.LoadConfigs import * #load_configs
from load.LoadParty import load_parties
from evaluates.MainTaskVFL import *
from utils.basic_functions import plot_model_performance
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def evaluate_no_attack(args):
    # No Attack
    set_seed(args.current_seed)

    vfl = MainTaskVFL(args)
    if args.dataset.dataset_name not in ['cora']:
        
        main_acc , stopping_iter, stopping_time, stopping_commu_cost= vfl.train()
    else:
        main_acc, stopping_iter, stopping_time = vfl.train_graph()

    # Save record 

    return vfl, main_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser("backdoor")
    parser.add_argument('--device', type=str, default='cuda', help='use gpu or cpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', type=int, default=97, help='random seed')
    parser.add_argument('--n_seeds', type=int, default=1, help='number of seeds')
    parser.add_argument('--configs', type=str, default='basic_configs', help='configure json file path')
    parser.add_argument('--save_model', type=bool, default=False, help='whether to save the trained model')
    args = parser.parse_args()


    accuracy_list = [] 


    config_file_path = './configs/'+args.configs+'.json'
    config_file = open(config_file_path,"r")
    config_dict = json.load(config_file)
    args_list = []

    args.data_split = None
    if not 'compare_centralized' in config_dict:
        config_dict['compare_centralized'] = False
    if not 'compare_single' in config_dict:
        config_dict['compare_single'] = False

    if config_dict['compare_centralized'] == True:
        assert config_dict['k'] != 1, "Comparing centralized training with centralized training"
        args_centralized = copy.deepcopy(args)

        input_dim , output_dim = 0, 0
        config_dict_centralized = copy.deepcopy(config_dict)
        for ik in range(config_dict['k']):
            input_dim += int(config_dict['model_list'][str(ik)]['input_dim'])
            output_dim += int(config_dict['model_list'][str(ik)]['output_dim'])
        config_dict_centralized['model_list']['0']['input_dim'] = input_dim
        config_dict_centralized['model_list']['0']['output_dim'] = output_dim
        #delete parties other than 0 in model_list
        for ik in range(1,config_dict['k']):
            del config_dict_centralized['model_list'][str(ik)]
        config_dict_centralized['k'] = 1
        config_dict_centralized['compare_centralized'] = False
        config_dict_centralized['compare_single'] = False


        args_centralized.case = 'centralized'

        args_list += load_basic_configs(config_dict_centralized, args_centralized)
    
    if config_dict['compare_single'] == True:
        assert config_dict['k'] != 1, "You can't compare single party training with single party training"
            
        input_dim , output_dim = 0, 0
        for ik in range(config_dict['k']):
            args_single = copy.deepcopy(args)
            args_single.data_split = [input_dim , input_dim + int(config_dict['model_list'][str(ik)]['input_dim'])]
            input_dim += int(config_dict['model_list'][str(ik)]['input_dim'])
            config_dict_single = copy.deepcopy(config_dict)
            
            config_dict_single['model_list']['0'] = config_dict['model_list'][str(ik)]
            for jk in range(1,config_dict['k']):
                del config_dict_single['model_list'][str(jk)]
            
            config_dict_single['k'] = 1
            config_dict_single['compare_single'] = False
            config_dict_single['compare_centralized'] = False

            #debug
            args_single.case = 'single' + str(ik)

            args_list += load_basic_configs(config_dict_single, args_single)
    
    #debug
    args.case = 'normal'
    args_list += load_basic_configs(config_dict, args)
    
    loss_accuracy_list = []
    accuracy_list = []

    for arg in args_list:
        accuracy = []
        loss_accuracy = []
        for seed in range(arg.seed, arg.seed+arg.n_seeds):
            arg.current_seed = seed
            set_seed(seed)
            print('================= iter seed ',seed,' =================')
            if arg.device == 'cuda':
                cuda_id = arg.gpu
                torch.cuda.set_device(cuda_id)
                print(f'running on cuda{torch.cuda.current_device()}')
            else:
                print('running on cpu')

            print('============ apply_trainable_layer=',arg.apply_trainable_layer,'============')
            #print('================================')
            
            print('case : ' + arg.case)
            
            # Save record for different defense method
            arg.exp_res_dir = f'exp_result/{arg.dataset}/Q{str(arg.Q)}/{str(arg.apply_trainable_layer)}/'
            if not os.path.exists(arg.exp_res_dir):
                os.makedirs(arg.exp_res_dir)
            filename = f'model={arg.model_list[str(0)]["type"]}.txt'
            arg.exp_res_path = arg.exp_res_dir + filename
            print(arg.exp_res_path)
            print('=================================\n')

            iterinfo='===== iter '+str(seed)+' ===='

            arg.basic_vfl = None
            arg.main_acc = None

            if arg.dataset == 'satellite': # train test splitting of POIs
                data = pd.read_csv(arg.data_root + r'\metadata.csv')
                #rename 1st column to 'POI'
                data = data.rename(columns={data.columns[0]: 'POI'})
                data = data.loc[data['POI'] != 'ASMSpotter-1-1-1']
                #split the POI between train and test
                POI = data['POI'].unique()

                arg.train_POI, arg.test_POI = train_test_split(POI, test_size=0.2, random_state=arg.current_seed, shuffle=True)
                arg.train_POI, arg.test_POI = set(arg.train_POI), set(arg.test_POI)
            
            arg = load_parties(arg)

            commuinfo='== commu:'+arg.communication_protocol

            arg.basic_vfl, arg.main_acc = evaluate_no_attack(arg)
            loss_accuracy.append([arg.basic_vfl.loss[-1], arg.basic_vfl.train_acc[-1], arg.basic_vfl.test_acc[-1], arg.basic_vfl.test_auc[-1]])
            accuracy.append(arg.basic_vfl.test_acc)
        loss_accuracy = np.mean(np.array(loss_accuracy), axis=0)
        loss_accuracy_list.append(loss_accuracy)
        accuracy_list.append(accuracy)
    

    
    print('================= Final Results averaged over seeds =================')
    for i in range(len(args_list)):
        print('case {} \t train_loss:{:.4f} train_acc:{:.4f} test_acc:{:.4f} test_auc:{:.4f}'.format(
            args_list[i].case, loss_accuracy_list[i][0], loss_accuracy_list[i][1], loss_accuracy_list[i][2], loss_accuracy_list[i][3]))
    

    cases_list = [arg.case for arg in args_list]
    plot_model_performance(cases_list, np.array(accuracy_list) )
