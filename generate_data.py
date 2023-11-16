import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class Stock(Dataset):
    def __init__(self,
                 path='E:Data/AMZN_10.csv',
                 sequence_length = 60,
                 future_time_steps = 14,
                 idx = None,
                 isVal = False,
                 includeInput = True,
                 ):
    
        self.df = pd.read_csv(path)
        self.sequence_length = sequence_length
        self.future_time_steps = future_time_steps
        
        x, y = self.init_dataset(includeInput)
        
        if idx != None:
            if isVal:
                x = x[idx:]
                y = y[idx:]
            else:
                x = x[:idx]
                y = y[:idx]
        
        self.X = self.normalize(x)
        self.y = self.normalize(y)
        
        
    def init_dataset(self, includeInput):
        time_series = self.preprocess(self.df)
        x = []
        y = []
        
        for i in range(len(time_series) - self.sequence_length - self.future_time_steps + 1):
            # Input data (past values)
            x.append(time_series[i : i + self.sequence_length])

            # Target data (future values)
            if includeInput == True:
                y.append(time_series[i : i + self.sequence_length + self.future_time_steps])
            else:
                y.append(time_series[i + self.sequence_length : i + self.sequence_length + self.future_time_steps])

        x = np.stack(x)
        y = np.stack(y)

        # Convert input and target data to PyTorch tensors
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        
        x = x.view(len(x), 1, -1)  # Shape: (batch_size, input_size, sequence_length)
        if includeInput == True:
            y = y.view(len(y), self.sequence_length + self.future_time_steps)  # Shape: (batch_size, self.sequence_length + future_time_steps)
        else:
            y = y.view(len(y), self.future_time_steps)  # Shape: (batch_size, future_time_steps)

        
        return x, y
    def preprocess(self, data):
        data['Date'] = pd.to_datetime(data['Date'])
        # order the values, even if they are already
        data =  data.sort_values(by='Date', ascending=True)
        
        time_series = data['Close'].values
        
        return time_series
        
    def normalize(self, data):
        min_value = data.min()
        max_value = data.max()
    
        normalized_array = 2 * (data - min_value) / (max_value - min_value) - 1
    
        return normalized_array
        
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        X = self.X[index]
        y = self.y[index]

        return X, y