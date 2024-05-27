import sys, os
sys.path.append(os.pardir)
import math
import json
import argparse
import copy
from models.autoencoder import AutoEncoder


communication_protocol_list = ['FedSGD','FedBCD_p','FedBCD_s','CELU','Quantization','Topk']

def load_basic_configs(config_dict, args):
    # print(config_dict)
    
    # args.main_lr, learning rate for main task
    args.main_lr = config_dict['lr'] if('lr' in config_dict) else 0.001
    assert (args.main_lr>0), "main learning rate should be >0"

    # args.main_epochs, iterations for main task
    args.main_epochs = config_dict['epochs'] if('epochs' in config_dict) else 50
    
    # args.early_stop_threshold, early stop max epoch
    args.early_stop_threshold = config_dict['early_stop_threshold'] if('early_stop_threshold' in config_dict) else 5
    
    # args.k, number of participants
    args.k = config_dict['k'] if('k' in config_dict) else 2
    assert (args.k % 1 == 0 and args.k>0), "k should be positive integers"

    # args.batch_size for main task
    args.batch_size = config_dict['batch_size'] if ('batch_size' in config_dict) else 2048
    
    # Communication Protocol
    communication_protocol_dict = config_dict['communication'] if ('communication' in config_dict) else None
    
    if communication_protocol_dict is not None:
        args.communication_protocol = communication_protocol_dict['communication_protocol'] if ('communication_protocol' in communication_protocol_dict) else 'FedBCD_p'
    else:
        args.communication_protocol = 'FedBCD_p'

    assert (args.communication_protocol in communication_protocol_list), "communication_protocol not available"
    
    args.Q = communication_protocol_dict['iteration_per_aggregation'] if ('iteration_per_aggregation' in communication_protocol_dict) else 1
    assert (args.Q % 1 == 0 and args.Q>0), "iteration_per_aggregation should be positive integers"
    
    args.quant_level = communication_protocol_dict['quant_level'] if ('quant_level' in communication_protocol_dict) else 0
    args.vecdim = communication_protocol_dict['vecdim'] if ('vecdim' in communication_protocol_dict) else 1
    args.num_update_per_batch = communication_protocol_dict['num_update_per_batch'] if ('num_update_per_batch' in communication_protocol_dict) else 5
    args.num_batch_per_workset = communication_protocol_dict['num_batch_per_workset'] if ('num_batch_per_workset' in communication_protocol_dict) else 5
    args.smi_thresh = communication_protocol_dict['smi_thresh'] if ('smi_thresh' in communication_protocol_dict) else 0.5
    
    if args.quant_level > 0:
        args.ratio = math.log(args.quant_level,2)/32
    args.ratio = communication_protocol_dict['ratio'] if ('ratio' in communication_protocol_dict) else 0.5
    print('Topk Ratio:',args.ratio)
    


    if args.communication_protocol == 'FedSGD':
        args.Q = 1
    
    print('communication_protocol:',args.communication_protocol)

    
    args.attacker_id = []
    # # args.early_stop, if use early stop
    # args.main_early_stop = config_dict['main_early_stop'] if ('main_early_stop' in config_dict) else 0
    # args.main_early_stop_param = config_dict['main_early_stop_param'] if ('main_early_stop_param' in config_dict) else 0.0001
    # # args.num_exp number of repeat experiments for main task
    # args.num_exp = config_dict['num_exp'] if ('num_exp' in config_dict) else 10
    
    # args.dataset_split
    args.dataset_split = config_dict['dataset'] if('dataset' in config_dict) else None
    args.num_classes = args.dataset_split['num_classes'] if('num_classes' in args.dataset_split) else 10

    # args.model_list, specify the types of models
    if 'model_list' in config_dict:
        config_model_dict = config_dict['model_list']
        #print('config_model_dict:',(len(config_model_dict)-2))
        assert ((len(config_model_dict)-2)==args.k), 'please alter party number k, model number should be equal to party number'
        
        model_dict = {}
        default_dict_element = {'type': 'MLP2', 'path': 'random_14*28_10', 'input_dim': 392, 'output_dim': 10}
        for ik in range(args.k):
            if str(ik) in config_model_dict:
                if 'type' in config_model_dict[str(ik)]:
                    if 'path' in config_model_dict[str(ik)] or (('input_dim' in config_model_dict[str(ik)]) and ('output_dim' in config_model_dict[str(ik)])):
                        model_dict[str(ik)] = config_model_dict[str(ik)]
                    else:
                        model_type_name = config_model_dict[str(ik)]['type']
                        temp = {'type':model_type_name, 'path':'../models/'+model_type_name+'/random'}
                        model_dict[str(ik)] = temp
                else:
                    model_dict[str(ik)] = default_dict_element
            else:
                model_dict[str(ik)] = default_dict_element
        args.model_list = model_dict
        args.apply_trainable_layer = config_model_dict['apply_trainable_layer'] if ('apply_trainable_layer' in config_model_dict) else 0
        args.global_model = config_model_dict['global_model'] if ('global_model' in config_model_dict) else 'ClassificationModelHostHead'
    else:
        default_model_dict = {}
        default_dict_element = {'type': 'MLP2', 'path': '../models/MLP2/random'}
        for ik in range(args.k):
            default_model_dict[str(ik)] = default_dict_element
        args.model_list = default_model_dict
        args.apply_trainable_layer = 0
        args.global_model = 'ClassificationModelHostHead'

    return [args]

    '''
    load attack[index] in attack_list
    '''
    config_file_path = './configs/'+config_file_name+'.json'
    config_file = open(config_file_path,"r")
    config_dict = json.load(config_file)

    args.attack_type = None
    args.apply_backdoor = False # replacement backdoor attack
    args.apply_nl = False # noisy label attack
    args.apply_ns = False # noisy sample attack
    args.apply_mf = False # missing feature attack
    
    # No Attack
    if index == -1:
        print('No Attack==============================')
        args.attack_name='No_Attack'
        args.attack_param_name = 'None'
        args.attack_param = None
        return args
    
    # init args about attacks
    assert args.apply_attack == True 
    # choose attack[index]
    attack_config_dict = config_dict['attack_list'][str(index)]

    args.attaker_id = attack_config_dict['party'] if('party' in attack_config_dict) else []

    if 'name' in attack_config_dict:
        args.attack_name = attack_config_dict['name']
        args.attack_configs = attack_config_dict['parameters'] if('parameters' in attack_config_dict) else None
        
        if args.attack_name in TARGETED_BACKDOOR:
            args.attack_type = 'targeted_backdoor'
            if 'backdoor' in args.attack_name.casefold():
                args.apply_backdoor = True
            
        elif args.attack_name in UNTARGETED_BACKDOOR:
            args.attack_type = 'untargeted_backdoor'
            if 'noisylabel' in args.attack_name.casefold():
                args.apply_nl = True
            if 'noisysample' in args.attack_name.casefold():
                args.apply_ns = True
            if 'missingfeature' in args.attack_name.casefold():
                args.apply_mf = True

        elif args.attack_name in LABEL_INFERENCE:
            args.attack_type = 'label_inference'

        elif args.attack_name in ATTRIBUTE_INFERENCE:
            args.attack_type = 'attribute_inference'

        elif args.attack_name in FEATURE_INFERENCE:
            args.attack_type = 'feature_inference'

        else:
            assert 0 , 'attack type not supported'
        
        if args.attack_name == 'NoisyLabel':
            args.attack_param_name = 'noise_type'
            args.attack_param = str(attack_config_dict['parameters']['noise_type'])+'_'+str(attack_config_dict['parameters']['noise_rate'])
        elif args.attack_name == 'MissingFeature':
            args.attack_param_name = 'missing_rate'
            args.attack_param = str(attack_config_dict['parameters']['missing_rate'])
        elif args.attack_name == 'BatchLabelReconstruction':
            args.attack_param_name = 'attack_lr'
            args.attack_param = str(attack_config_dict['parameters']['lr'])
        elif args.attack_name == 'NoisySample':
            args.attack_param_name = 'noise_lambda'
            args.attack_param = str(attack_config_dict['parameters']['noise_lambda'])
        else:
            args.attack_param_name = 'None'
            args.attack_param = None
        
    else:
        assert 'name' in attack_config_dict, "missing attack name"

    # Check: Centralized Training
    if args.k ==1:
        print('k=1, Launch Centralized Training, All Attack&Defense dismissed, Q set to 1')
        args.apply_attack = False # bli/ns/ds attack
        args.apply_backdoor = False # replacement backdoor attack
        args.apply_nl = False # noisy label attack
        args.apply_ns = False # noisy sample attack
        args.apply_mf = False # missing feature attack
        args.apply_defense = False
        args.apply_mid = False
        args.apply_cae = False
        args.apply_dcae = False
        args.apply_dp = False
        args.Q=1

    return args
