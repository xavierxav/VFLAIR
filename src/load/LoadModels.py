import sys, os
sys.path.append(os.pardir)

from models.global_models import *
from models.mlp import *
from models.cnn import *

def load_basic_models(args,index):
    current_model_type = args.model_list[index]['type']
    print(f"current_model_type={current_model_type}")

    local_model = globals()[current_model_type](args.model_list[index])
    local_model = local_model.to(args.runtime.device)
    print(f"local_model parameters: {sum(p.numel() for p in local_model.parameters())}")

    local_model_optimizer = torch.optim.Adam(list(local_model.parameters()), lr=args.lr, weight_decay= args.model_list[index].weight_decay)
        
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

    return local_model, local_model_optimizer, global_model, global_model_optimizer
