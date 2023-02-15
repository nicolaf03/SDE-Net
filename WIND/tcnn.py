import torch
import torch.nn as nn

class TCN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, kernel_size):
        super(TCN, self).__init__()
        
        self.tcn = nn.Sequential()
        for i in range(num_layers):
            self.tcn.add_module('conv_layer_{}'.format(i), 
                                nn.Conv1d(input_size if i == 0 else hidden_size, 
                                          hidden_size, kernel_size, padding=(kernel_size-1)//2))
            self.tcn.add_module('batch_norm_{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.tcn.add_module('relu_{}'.format(i), nn.ReLU())
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch_size, sequence_length, input_size) -> (batch_size, input_size, sequence_length)
        x = self.tcn(x)
        x = x[:, :, -1]
        x = self.fc(x)
        return x

input_size = 1 # the number of input features (e.g., 1 for univariate time-series)
output_size = 2 # the number of outputs (e.g., 2 for drift and diffusion coefficients of SDE)
hidden_size = 128
num_layers = 5
kernel_size = 3

model = TCN(input_size, output_size, hidden_size, num_layers, kernel_size)

# Define the loss function, e.g., mean squared error between the predicted and actual coefficients
criterion = nn.MSELoss()

# Define the optimizer, e.g., Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
# Train the model
for epoch in range(num_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        outputs = model(data)
        test_loss = criterion(outputs, target)
