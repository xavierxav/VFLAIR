import torch.optim as optim
from dataset.party_dataset import ActiveSatelliteDataset, SatteliteLABELS
from torch.utils.data import DataLoader
from models.cnn import SatelliteCNN
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn.init as init
from torchsummary import summary
import torch



def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def training():
    # Hyperparameters
    batch_size = 128
    learning_rate = 1e-3
    num_epochs = 5
    data_root = "C:/Users/XD278777/Desktop/worldstrat/dataset"

    data = pd.read_csv(data_root + r'\metadata.csv')
    # Rename 1st column to 'POI'
    data = data.rename(columns={data.columns[0]: 'POI'})
    data = data.loc[data['POI'] != 'ASMSpotter-1-1-1']
    # Split the POI between train and test
    POI = data['POI'].unique()

    train_POI, test_POI = train_test_split(POI, test_size=0.2, random_state=45)

    train_dataset = ActiveSatelliteDataset(test_train_POI=train_POI, index=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Model, Loss, Optimizer
    model = SatelliteCNN(num_classes=len(SatteliteLABELS)).cuda()
    model.apply(initialize_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

if __name__ == '__main__':
    # Assuming your model class is defined as SatelliteCNN
    model = SatelliteCNN(num_classes=len(SatteliteLABELS)).cuda()

    # Print the model summary
    summary(model, input_size=(12, 160, 160))
    training()
