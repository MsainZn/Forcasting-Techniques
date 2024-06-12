import torch
import torch.nn as nn

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Input layer to hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)  # Hidden layer to output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.3):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # BiLSTM Layer
        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True, 
                            bidirectional=True, 
                            dropout=dropout)
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        # Initial hidden state and cell state
        # The number of layers in the LSTM multiplied by 2 because it's a bidirectional LSTM (one direction for forward pass and one for backward).
        # x.size(0): The batch size, which is the number of sequences in the input batch.
        # self.hidden_size: The number of hidden units in each LSTM cell.
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size * 2)
        
        # Select the last time step output for each sequence in the batch
        out = out[:, -1, :]  # out: tensor of shape (batch_size, hidden_size * 2)
        
        # Pass through the fully connected layer
        out = self.fc(out)  # out: tensor of shape (batch_size, output_size)
        
        return out