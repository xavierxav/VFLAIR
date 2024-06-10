import torch
from torch.utils.data import Dataset
import pandas as pd
import tifffile as tiff
import os
from torchvision import transforms


SatteliteLABELS = {'Amnesty POI' : 0 , 'ASMSpotter' : 1 , 'Landcover' : 2 , 'UNHCR' : 3}
MeanSatellite = [0.08366308, 0.0975707, 0.12684818, 0.14296457, 0.17918019, 0.24204625, 0.26691017, 0.27571142, 0.2840153, 0.28720635, 0.2510504, 0.19122222]
StdSatellite = [21.940327, 22.558502, 22.246166, 23.915598, 23.219856, 20.530775, 20.111584, 20.508501, 19.583115, 20.758244, 20.116787, 19.624557]

normalize = transforms.Normalize(mean=MeanSatellite, std=StdSatellite)

class PassiveDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_idx):
        data_i= self.data[item_idx]
        return torch.tensor(data_i, dtype=torch.float32), 0


class ActiveDataset(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_idx):
        data_i, target_i = self.data[item_idx], self.labels[item_idx]
        return torch.tensor(data_i, dtype=torch.float32), torch.tensor(target_i, dtype=torch.long)


class SatelliteDataset(Dataset):
    def __init__(self, test_train_POI , csv_file = r'\metadata.csv', index = 0, root = r'C:\Users\XD278777\Desktop\worldstrat\dataset', transform=normalize):
        
        self.index = index
        self.root = os.path.normpath(root)
        self.transform = transform

        self.data = self.metadata( os.path.normpath(csv_file) , test_train_POI)

    def __len__(self):
        return len(self.data)
    
    def metadata(self, csv_file, test_train_POI, cloud_cover = False):
        data = pd.read_csv(self.root + csv_file)
        
        # Rename 1st column to 'POI'
        data = data.rename(columns={data.columns[0]: 'POI'})
        
        # Filter by POI
        data = data.loc[data['POI'].isin(test_train_POI)]
        
        if cloud_cover:
            # Sort data by 'cloud_cover' within each 'POI'
            data = data.sort_values(by=['POI', 'cloud_cover'])
        else:
            # Sort data by 'n' within each 'POI'
            data = data.sort_values(by=['POI', 'n'])
        
        # Group by 'POI' and pick the appropriate row based on self.index
        def pick_image(group):
            if self.index is not None:

                return group.iloc[self.index]
            else:
                return group  # Return all rows if self.index is None
        
        # Apply the pick_image function to each group
        data = data.groupby('POI').apply(pick_image).reset_index(drop=True)
        
        # Ensure 'POI' column is ordered according to test_train_POI
        data['POI'] = pd.Categorical(data['POI'], categories=test_train_POI, ordered=True)
        
        # Sort data by 'POI' to maintain the order specified in test_train_POI
        data = data.sort_values('POI')
        
        return data.reset_index(drop=True)



    def __getitem__(self, idx):
        img_path = self.root + '\\lr_dataset\\' + self.data.at[idx, 'POI'] + '\L2A\\' + self.data.at[idx, 'POI'] + '-' + str(self.data.at[idx, 'n']) + '-' + 'L2A_data.tiff'
        image = torch.tensor(tiff.imread(img_path), dtype=torch.float32)
        if image.shape[-1] != 12:
            raise ValueError("Expected image with 12 channels")
        image = image.permute(2, 0, 1)
        if self.transform is not None:
            image = self.transform(image)

        return image , 0
    
class ActiveSatelliteDataset(SatelliteDataset):
    def __init__(self, test_train_POI , csv_file = r'\metadata.csv', index = 0, root = r'C:\Users\XD278777\Desktop\worldstrat\dataset', transform=None):
        super().__init__(test_train_POI, csv_file, index, root, transform)
        self.labels = self.data['POI'].apply(lambda x: SatteliteLABELS[x.split('-')[0]])
        
    def __getitem__(self, idx):
        image = super().__getitem__(idx)[0]
        label = self.labels[idx]
        return image, label