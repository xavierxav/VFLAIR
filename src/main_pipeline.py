import os
import numpy as np
import pandas as pd

import random
import torch
from load.LoadParty import load_parties
from evaluates.MainTaskVFL import *
from utils.basic_functions import plot_model_performance
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

import hydra
from omegaconf import DictConfig , OmegaConf
from src.config.config_models import Config

def set_seed(seed=0):
    """
    Set the seed for reproducibility across various libraries.
    
    Parameters:
    seed (int): The seed value to use for random number generation.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(config_path="config", config_name="credit_default_config")
def main(cfg: DictConfig):
    # Validate the configuration using Pydantic
    config = Config(**OmegaConf.to_container(cfg, resolve=True))

    # Load and process configurations
    args_list = load_configs(config)

    # Your existing logic to handle args_list
    process_args_list(args_list)

def load_configs(cfg: Config):
    args_list = []

    # Normal case
    cfg.case = 'normal'
    args_list.append(cfg)

    # Centralized case
    if cfg.compare_centralized:
        centralized_cfg = copy.deepcopy(cfg)
        input_dim, output_dim = 0, 0
        for ik in range(cfg.k):
            input_dim += cfg.model_list[ik].input_dim
            output_dim += cfg.model_list[ik].output_dim
        centralized_cfg.model_list = centralized_cfg.model_list[0]
        centralized_cfg.model_list.input_dim = input_dim
        centralized_cfg.model_list.output_dim = output_dim
        centralized_cfg.k = 1
        centralized_cfg.case = 'centralized'
        args_list.append(centralized_cfg)

    # Single party case
    if cfg.compare_single:
        input_dim, output_dim = 0, 0
        for ik in range(cfg.k):
            single_cfg = copy.deepcopy(cfg)
            single_cfg.model_list = [single_cfg.model_list[ik]]
            single_cfg.k = 1
            single_cfg['case'] = f'single_party_{ik}'
            single_cfg['data_split'] = [input_dim, input_dim + single_cfg.model_list[0].input_dim]
            single_cfg['index'] = ik
            input_dim += single_cfg.model_list[0].input_dim
            args_list.append(single_cfg)

    return args_list

#TODO: Add the logic to process the args_list
def process_args_list(args_list):
    loss_accuracy_list = []
    accuracy_list = []
    for args in args_list:
        all_accuracy = []
        all_loss_accuracy = []
        for seed in range(args.runtime.seed, args.runtime.seed + args.runtime.n_seeds):
            loss_accuracy, accuracy = process_seed_iteration(args, seed)
            all_loss_accuracy.extend(loss_accuracy)
            all_accuracy.extend(accuracy)
        
        mean_loss_accuracy = np.mean(np.array(all_loss_accuracy), axis=0)
        loss_accuracy_list.append(mean_loss_accuracy)
        accuracy_list.append(all_accuracy)

    print('================= Final Results averaged over seeds =================')
    for i, args in enumerate(args_list):
        print(f'case {args["case"]} \t train_loss:{loss_accuracy_list[i][0]:.4f} \t train_acc:{loss_accuracy_list[i][1]:.4f} \t test_acc:{loss_accuracy_list[i][2]:.4f} \t test_auc:{loss_accuracy_list[i][3]:.4f}')
    
    cases_list = [args['case'] for args in args_list]
    plot_model_performance(cases_list, np.array(accuracy_list))

def process_seed_iteration(args, seed):
    accuracy = []
    loss_accuracy = []

    args.runtime.current_seed = seed
    set_seed(seed)
    print(f'================= iter seed {seed} =================')
    if args.runtime.device == 'cuda':
        torch.cuda.set_device(args.runtime.gpu)
        print(f'running on cuda {torch.cuda.current_device()}')
    else:
        print('running on cpu')

    print(f'============ apply_trainable_layer={args.global_model.apply_trainable_layer} ============')
    print(f'case : {args.case}')

    if args.dataset.dataset_name == 'satellite': # train test splitting of POIs
        data = pd.read_csv(os.path.join(args.dataset.data_root, 'satellite_dataset' , 'metadata.csv'))
        # rename 1st column to 'POI'
        data = data.rename(columns={data.columns[0]: 'POI'})
        data = data.loc[data['POI'] != 'ASMSpotter-1-1-1']
        # split the POI between train and test
        POI = data['POI'].unique()

        args.dataset.train_POI, args.dataset.test_POI = train_test_split(POI, test_size=0.2, random_state=args.runtime.current_seed, shuffle=True)
        args.dataset.train_POI, args.dataset.test_POI = set(args.dataset.train_POI), set(args.dataset.test_POI)
    
    args = load_parties(args)

    # Evaluate the model and collect results
    set_seed(args.runtime.current_seed)

    vfl = MainTaskVFL(args)
    vfl.train()

    loss_accuracy.append([
        vfl.loss[-1], 
        vfl.train_acc[-1], 
        vfl.test_acc[-1], 
        vfl.test_auc[-1]
    ])
    accuracy.append(vfl.test_acc)
    
    return loss_accuracy, accuracy

if __name__ == '__main__':
    main()