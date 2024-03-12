import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import numpy as np
from itertools import product
import pandas as pd
import sys
import os
import csv

class CognitiveDataset(Dataset):
    def __init__(self):
        self.folder = sys.argv[1] # path to data folder ex)"../sample_dataset/"
        self.X_file_name = sys.argv[2] #feature file: ex) "lowlevel.csv"
        self.y_file_name = sys.argv[3] #file name of label ex)"mental_demand.npy"
        self.X_paths = self.file_name_list(self.X_file_name)
        self.y_paths = self.file_name_list(self.y_file_name)
        self.X_max_len = self.get_max_size()
        print(f"X_max_len: {self.X_max_len}")
    
    def __getitem__(self,idx):
        data_point = pd.read_csv(self.X_paths[idx]).replace([np.nan,np.inf, -np.inf], 0).values.flatten()
        padded_data = np.zeros(self.X_max_len)
        print(f"zero padded: {self.X_max_len-len(data_point)} values")
        padded_data[:len(data_point)] = data_point #zero pad
        data_point = torch.tensor(padded_data)
        label =  0 if np.load(self.y_paths[idx]) < 11 else 1
        label = torch.tensor(label)
        return data_point, label
    
    def __len__(self):
        return len(self.X_paths)

    """Helper functions"""
    def get_max_size(self):
        max_len = 0
        for file in self.X_paths:
            df = pd.read_csv(file).replace([np.nan, np.inf, -np.inf], 0)
            if len(df.values.flatten()) > max_len:
                max_len = len(df.values.flatten())
        return max_len
    
    def file_name_list(self,target):
        """
        param: 
        - target(what type of file we want) ex) mental_demand.npy
        returns the list of files we want to find
        """
        target_files = []
        for dirpath, _, filenames in os.walk(self.folder):
            for fN in filenames:
                if fN.endswith(target):
                    file_path = os.path.join(dirpath, fN)
                    target_files.append(file_path)
        return sorted(target_files)
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size_1=50, hidden_size_2=5, output_size=1):
        super().__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        self.layers = nn.Sequential(
            nn.Linear(input_size, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.hidden_size_2),
            nn.ReLU(),
            nn.Linear(self.hidden_size_2, self.output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

def main():
    #hyperparameters
    batch_size = 4
    learning_rate = 1e-4
    epochs = 10

    #fixed random seed
    torch.manual_seed(42)
    
    dataset = CognitiveDataset()

    test_prop = 0.3
    test_size = int(dataset.__len__() * test_prop)
    train_set, test_set = torch.utils.data.random_split(dataset, [
            (dataset.__len__() - (test_size)), 
            test_size
    ])

    train_dataloader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
    )

    test_dataloader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=True,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = MLP(len(dataset[0][0])).to(device)
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    for epoch in range(epochs):
        print(f'Entering Epoch {epoch+1}')
        current_loss = 0.0

        #iterate over data
        for i, (input, target) in enumerate(train_dataloader):
            input = input.float().to(device)
            target = target.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            output = model(input)

            loss = loss_function(output, target)

            loss.backward()

            optimizer.step()

            #stats
            current_loss += loss.item()
            print(f'Loss after mini-batch {i+1}: {current_loss}')
            current_loss = 0.0
       
    np.save('train_loss.npy',train_losses)
    print("Training has completed")

    model_path = os.path.join(os.getcwd(),"model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


    """testing"""
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for input,target in test_dataloader:
            input = input.float().to(device)
            target = target.float().unsqueeze(1).to(device)

            output = model(input) #forward pass

            #loss
            loss = loss_function(output, target)
            test_loss += loss.item()
        
    test_loss = test_loss / len(test_dataloader)
    print(f"Testset loss: {test_loss:5f}")

    test_loss = np.array(test_loss)
    np.save('test_loss.npy',test_loss)


if __name__ == '__main__':
    main()

    """
    dataset = CognitiveDataset()
    for i in range (len(dataset)):
        print(dataset[i][0].shape,dataset[i][1])
    """