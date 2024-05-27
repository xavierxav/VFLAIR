import sys, os
sys.path.append(os.pardir)

import argparse
import numpy as np
import pickle

from models.bottom_models import *
from models.global_models import *
from models.autoencoder import *

def create_model(bottom_model, ema=False, size_bottom_out=10, num_classes=10):
    model = BottomModelPlus(bottom_model,size_bottom_out, num_classes,
                                num_layer=2,
                                activation_func_type='ReLU',
                                use_bn=0)
    model = model

    if ema:
        for param in model.parameters():
            param.detach_()

    return model


def load_models(args):
    args.net_list = [None] * args.k
    for ik in range(args.k):
        current_model_type = args.model_list[str(ik)]['type']
        current_model_path = args.model_list[str(ik)]['path']
        args.net_list[ik] = pickle.load(open('.././src/models/model_parameters/'+current_model_type+'/'+current_model_path+'.pkl',"rb"))
        args.net_list[ik] = args.net_list[ik].to(args.device)
    # important
    return args

def load_basic_models(args,index):
    current_model_type = args.model_list[str(index)]['type']
    print(f"current_model_type={current_model_type}")
    current_input_dim = args.model_list[str(index)]['input_dim'] if 'input_dim' in args.model_list[str(index)] else args.half_dim[index]
    current_hidden_dim = args.model_list[str(index)]['hidden_dim'] if 'hidden_dim' in args.model_list[str(index)] else -1
    current_output_dim = args.model_list[str(index)]['output_dim']
    current_vocab_size = args.model_list[str(index)]['vocab_size'] if 'vocab_size' in args.model_list[str(index)] else -1
    current_lr = args.model_list[str(index)]['lr'] if 'lr' in args.model_list[str(index)] else args.main_lr

    if 'resnet' in current_model_type.lower() or 'lenet' in current_model_type.lower() or 'cnn' in current_model_type.lower() or 'alexnet' in current_model_type.lower():
        local_model = globals()[current_model_type](current_output_dim)
    elif 'gcn' in current_model_type.lower():
        local_model = globals()[current_model_type](nfeat=current_input_dim,nhid=current_hidden_dim,nclass=current_output_dim, device=args.device, dropout=0.0, lr=current_lr)
    elif 'lstm' in current_model_type.lower(): 
        local_model = globals()[current_model_type](current_vocab_size, current_output_dim)
    else:
        local_model = globals()[current_model_type](current_input_dim, current_output_dim)
    local_model = local_model.to(args.device)
    print(f"local_model parameters: {sum(p.numel() for p in local_model.parameters())}")
    local_model_optimizer = torch.optim.Adam(list(local_model.parameters()), lr=current_lr, weight_decay=0.0)
        
    global_model = None
    global_model_optimizer = None
    if index == args.k-1:
        if args.apply_trainable_layer == 0:
            global_model = globals()[args.global_model]()
            global_model = global_model.to(args.device)
            global_model_optimizer = None
        else:
            print("global_model", args.global_model)
            global_input_dim = 0
            for ik in range(args.k):
                global_input_dim += args.model_list[str(ik)]['output_dim']
            global_model = globals()[args.global_model](global_input_dim, args.num_classes)
            global_model = global_model.to(args.device)
            global_model_optimizer = torch.optim.Adam(list(global_model.parameters()), lr=args.main_lr)
            # print(f"use SGD for global optimizer for PMC checking")
            # global_model_optimizer = torch.optim.SGD(list(global_model.parameters()), lr=args.main_lr, momentum=0.9, weight_decay=5e-4)

    return args, local_model, local_model_optimizer, global_model, global_model_optimizer

def load_defense_models(args, index, local_model, local_model_optimizer, global_model, global_model_optimizer):
    print('Load Defense models')
    # no defense at all, set some variables as None
    args.encoder = None
    return args, local_model, local_model_optimizer, global_model, global_model_optimizer

def load_models_per_party(args, index):
    args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_basic_models(args,index)
    args, local_model, local_model_optimizer, global_model, global_model_optimizer = load_defense_models(args, index, local_model, local_model_optimizer, global_model, global_model_optimizer)
    # important
    return args, local_model, local_model_optimizer, global_model, global_model_optimizer
