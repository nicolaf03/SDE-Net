import numpy as np
import torch

class DataLoader:
    def __init__(self, data, train_ratio):
        self.data = data
        self.train_ratio = train_ratio
        
    def split_data(self):
        train_size = int(len(self.data) * self.train_ratio)
        train_data = self.data[:train_size]
        test_data = self.data[train_size:]
        return train_data, test_data
    
    def get_inputs_targets(self, data, sequence_length):
        inputs = []
        targets = []
        for i in range(len(data) - sequence_length):
            inputs.append(data[i:i+sequence_length])
            targets.append(data[i+sequence_length])
        return inputs, targets
    
    def load_data(self, sequence_length):
        train_data, test_data = self.split_data()
        train_inputs, train_targets = self.get_inputs_targets(train_data, sequence_length)
        test_inputs, test_targets = self.get_inputs_targets(test_data, sequence_length)
        train_inputs = torch.tensor(train_inputs).float()
        train_targets = torch.tensor(train_targets).float()
        test_inputs = torch.tensor(test_inputs).float()
        test_targets = torch.tensor(test_targets).float()
        return train_inputs, train_targets, test_inputs, test_targets
