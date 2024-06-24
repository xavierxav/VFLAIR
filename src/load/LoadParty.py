from party.passive_party import PassiveParty
from party.active_party import ActiveParty
from load.LoadDataset import load_dataset_per_party

def load_parties(args):
    
    train_datasets, test_datasets = load_dataset_per_party(args)
    args.parties = [PassiveParty(args, ik , train_datasets[ik], test_datasets[ik]) for ik in range(args.k-1)]
    args.parties += [ActiveParty(args, args.k-1, train_datasets[args.k-1], test_datasets[args.k-1])]

    return args

