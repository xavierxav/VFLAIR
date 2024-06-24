import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.dataset.party_dataset import SatteliteLABELS, SatelliteDataset, PassiveDataset, ActiveDataset, ActiveSatelliteDataset

def load_dataset_per_party(args):
    print('load_dataset_per_party')

    if args.dataset.dataset_name == 'credit':
        csv_file = os.path.join(args.dataset.data_root, "tabledata" , "UCI_Credit_Card.csv")
        df = pd.read_csv( csv_file )
        print("credit dataset loaded")

        X = df[["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]].values
        y = df["default.payment.next.month"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.runtime.seed, stratify=y)

        train_datasets = dataset_partition_credit(args, X_train)
        test_datasets = dataset_partition_credit(args, X_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)
        for i in range(len(train_datasets) - 1): # passive parties without labels
            scaler = StandardScaler()
            train_datasets[i] = scaler.fit_transform(train_datasets[i])
            test_datasets[i] = scaler.transform(test_datasets[i])
            train_datasets[i] = PassiveDataset(train_datasets[i])
            test_datasets[i] = PassiveDataset(test_datasets[i])
        # active party with labels
        scaler = StandardScaler()
        train_datasets[-1] = scaler.fit_transform(train_datasets[-1])
        test_datasets[-1] = scaler.transform(test_datasets[-1])
        train_datasets[-1] = ActiveDataset(data = train_datasets[-1], labels=y_train)
        test_datasets[-1] = ActiveDataset(data = test_datasets[-1], labels=y_test)

        return train_datasets, test_datasets

    if args.dataset.dataset_name == 'satellite':
        # Load the df
        csv_file = os.path.join(args.dataset.data_root, 'satellite_dataset', 'metadata.csv')
        df = pd.read_csv(csv_file)

        # Rename 1st column to 'POI'
        df = df.rename(columns={df.columns[0]: 'POI'})
        df = df.loc[df['POI'] != 'ASMSpotter-1-1-1']

        # Identify unique POIs
        unique_pois = df[['POI']].drop_duplicates()

        # Add a 'label' column based on POI
        unique_pois['label'] = unique_pois['POI'].apply(lambda x: SatteliteLABELS[x.split('-')[0]])

        # Split POIs into train and test sets with stratification by label
        train_pois, test_pois = train_test_split(unique_pois['POI'], test_size=0.2, random_state=args.runtime.seed, stratify=unique_pois['label'])

        train_datasets = dataset_partition_satellite(args, df, train_pois)
        test_datasets = dataset_partition_satellite(args, df, test_pois)
        for i in range(len(train_datasets) - 1): # passive parties without labels
            train_datasets[i] = SatelliteDataset(dataset_dict = args.dataset, metadata = train_datasets[i])
            test_datasets[i] = SatelliteDataset(dataset_dict = args.dataset, metadata = test_datasets[i])
        # active party with labels
        train_datasets[-1] = ActiveSatelliteDataset(dataset_dict = args.dataset, metadata = train_datasets[-1])
        test_datasets[-1] = ActiveSatelliteDataset(dataset_dict = args.dataset, metadata = test_datasets[-1])

        return train_datasets, test_datasets

def dataset_partition_credit(args, X ):
    if args.case[:12] == 'single_party':
        return [X[:, args.data_split[0]:args.data_split[1]]]
    elif args.k == 1:
        return [X]
    else:
        column = 0
        X_list = []
        for ik in range(args.k):
            input_dim = int(args.model_list[ik]['input_dim'])
            X_list.append(X[:, column:column + input_dim])
            column += input_dim
        return X_list
        
def dataset_partition_satellite(args, data, test_train_POI):

    # Filter by POI
    data = data.loc[data['POI'].isin(test_train_POI)]
    
    if args.dataset.cloud_cover_ranking:
        # Sort data by 'cloud_cover' within each 'POI'
        data = data.sort_values(by=['POI', 'cloud_cover'])
    else:
        # Sort data by 'n' within each 'POI'
        data = data.sort_values(by=['POI', 'n'])
    
    # Group by 'POI' and pick the appropriate row based on self.index
    def pick_image(group, index):
        return group.iloc[index]
    
    if args.case[:12] == 'single_party':
        data = data.groupby('POI').apply(pick_image, index=args.index).reset_index(drop=True)
        # Ensure 'POI' column is ordered according to test_train_POI
        data['POI'] = pd.Categorical(data['POI'], categories=test_train_POI, ordered=True)
        
        # Sort data by 'POI' to maintain the order specified in test_train_POI
        data = data.sort_values('POI')
        
        return [data.reset_index(drop=True)]
    elif args.k == 1:
        return [data]
    else :
        data_list = []
        for ik in range(args.k):
            data_ik = data.groupby('POI').apply(pick_image, index=ik).reset_index(drop=True)
            # Ensure 'POI' column is ordered according to test_train_POI
            data_ik['POI'] = pd.Categorical(data_ik['POI'], categories=test_train_POI, ordered=True)
            
            # Sort data by 'POI' to maintain the order specified in test_train_POI
            data_ik = data_ik.sort_values('POI')
            data_list.append(data_ik.reset_index(drop=True))
        return data_list
