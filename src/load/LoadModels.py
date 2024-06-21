import sys, os
sys.path.append(os.pardir)

from models.bottom_models import *
from models.global_models import *
from models.autoencoder import *

def load_basic_models(args,index):
    current_model_type = args.model_list[index]['type']
    print(f"current_model_type={current_model_type}")
    current_input_dim = args.model_list[index].input_dim if hasattr(args.model_list[index], 'input_dim') else args.half_dim[index]
    current_hidden_dim = args.model_list[index].hidden_dim if hasattr(args.model_list[index], 'hidden_dim') else -1
    current_output_dim = args.model_list[index].output_dim
    current_vocab_size = args.model_list[index].vocab_size if hasattr(args.model_list[index], 'vocab_size') else -1
    current_lr = args.model_list[index].lr if hasattr(args.model_list[index], 'lr') else args.lr

    if 'resnet' in current_model_type.lower() or 'lenet' in current_model_type.lower() or 'cnn' in current_model_type.lower() or 'alexnet' in current_model_type.lower():
        local_model = globals()[current_model_type](current_output_dim)
    elif 'gcn' in current_model_type.lower():
        local_model = globals()[current_model_type](nfeat=current_input_dim,nhid=current_hidden_dim,nclass=current_output_dim, device=args.runtime.device, dropout=0.0, lr=current_lr)
    elif 'lstm' in current_model_type.lower(): 
        local_model = globals()[current_model_type](current_vocab_size, current_output_dim)
    else:
        local_model = globals()[current_model_type](current_input_dim, current_output_dim)
    local_model = local_model.to(args.runtime.device)
    print(f"local_model parameters: {sum(p.numel() for p in local_model.parameters())}")
    local_model_optimizer = torch.optim.Adam(list(local_model.parameters()), lr=current_lr, weight_decay= args.model_list[index].weight_decay)
        
    global_model = None
    global_model_optimizer = None
    if index == args.k-1:
        if args.global_model.apply_trainable_layer == 0:
            global_model = globals()[args.global_model.model]()
            global_model = global_model.to(args.runtime.device)
            global_model_optimizer = None
        else:
            print("global_model", args.global_model.model)
            global_input_dim = 0
            for ik in range(args.k):
                global_input_dim += args.model_list[ik]['output_dim']
            global_model = globals()[args.global_model.model](global_input_dim, args.dataset.num_classes)
            global_model = global_model.to(args.runtime.device)
            global_model_optimizer = torch.optim.Adam(list(global_model.parameters()), lr=args.lr, weight_decay= args.global_model.weight_decay )
            # print(f"use SGD for global optimizer for PMC checking")
            # global_model_optimizer = torch.optim.SGD(list(global_model.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    return local_model, local_model_optimizer, global_model, global_model_optimizer
