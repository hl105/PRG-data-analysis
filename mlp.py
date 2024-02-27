import torch
from torch import nn
#from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import numpy as np
from itertools import product
import pandas as pd
import sys
import os

class PreProcessData():
    def __init__(self, path=""):
        self.path = sys.argv[1] # path to data folder ex)"../sample_dataset/"
        self.task = sys.argv[2] #task file: ex) "lowlevel.csv"
        self.label = sys.argv[3] #file name of label ex)"mental_demand.npy"
        self.X = None
        self.y = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


    def file_name_list(self,target):
        """
        param: 
        - target(what type of file we want) ex) mental_demand.npy
        returns the list of files we want to find
        """
        target_files = []
        for dirpath, _, filenames in os.walk(self.path):
            for fN in filenames:
                if fN.endswith(target):
                    file_path = os.path.join(dirpath, fN)
                    target_files.append(file_path)
        return sorted(target_files)
    
    def label_to_binary(self):
        """
        converts label (1-21) to binary values with 11 as cutoff
        saves all task labels to task_binary_label.csv
        """
        data = {"label":[]}
        for file in self.file_name_list(self.label):
            label = np.load(file)
            binary = 0 if label < 11 else 1
            data['label'].append(binary)
        df = pd.DataFrame(data)
        return df
    
    def tensor_labels(self):
        """
        creates tensor with labels
        """
        df =  self.label_to_binary()
        tensor = torch.tensor(df['label'].values.flatten(),dtype=torch.float32).to(self.device)
        self.y = tensor
        print("\nlabel tensor size:",tensor.size())
        return tensor
        
    def create_task_tensor(self,file):
        """
        param: task csv_file
        replace NaN values with 0 and creates a tensor
        """
        df = pd.read_csv(file)
        df = df.replace([np.nan,np.inf, -np.inf], 0)

        assert not df.isnull().values.any() #check if NaN val exists

        tensor = torch.tensor(df.values.flatten(),dtype=torch.float32).to(self.device)
        #print(file, tensor.size())
        return tensor
    
    def stack_task_tensors(self):
        """
        param: list of file names we want to use
        zero pad tensors to maintain same length
        aggregates tensors(flattened tasks) to a single dataset
        """
        tensors = []
        max_len = 0
        for file in self.file_name_list(self.task):
            ts = self.create_task_tensor(file)
            ts_size = ts.size(0)
            tensors.append(ts)
            if ts_size > max_len:
                max_len = ts_size

        zero_padded_tensors = []
        for ts in tensors:
            zero_padded_tensor = nn.functional.pad(input=ts, pad=(0,max_len-ts.size(0)))
            zero_padded_tensors.append(zero_padded_tensor)

        task_tensors = torch.stack(zero_padded_tensors)
        print("task tensor size", task_tensors.size())
        self.X = task_tensors
        return task_tensors
    

class MLP(nn.Module):
    def __init__(self,input_size=720192,hidden_size_1=1000, hidden_size_2=40, output_size=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2,output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
    
class TaskData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    #data preprocessing
    ppd = PreProcessData()
    ppd.label_to_binary()
    ppd.tensor_labels()
    ppd.stack_task_tensors()

    torch.manual_seed(42)
    X,y = ppd.X, ppd.y
    dataset = TaskData(X, y)

    #mlp
    def train(train_loader,model,loss_function, optimizer):
        model.train()

        total_loss = 0
        for input, target in train_loader:
            input, target = input.float(), target.float()
            target = target.unsqueeze(1)
            
            optimizer.zero_grad()

            output = model(input) #forward pass
            
            loss = loss_function(output, target)
            total_loss += loss

            #backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = total_loss/len(train_loader)
        print(f"Average loss: {train_loss:7f}")

    def test(test_loader,model,loss_function):
        model.eval()

        test_loss = 0
        with torch.no_grad():
            for input,target in test_loader:
                input, target = input.float(), target.float()
                target = target.unsqueeze(1)

                output = model(input) #forward pass

                #loss
                loss = loss_function(output, target)
                test_loss += loss.item()
        
        test_loss = test_loss / len(test_loader)
        print(f"Testset average loss: {test_loss:7f}")
    
    train_size = 12
    test_size = len(dataset) - train_size #18-12
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_batch_size = 2
    test_batch_size = 2

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MLP().to(device)

    loss_function = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    for epoch in range(0,epochs):
        print(f'Entering Epoch {epoch+1}')
        train(train_loader,model,loss_function, optimizer)

    print("Training has completed")

    test(test_loader,model,loss_function)


if __name__ == '__main__':
    main()






