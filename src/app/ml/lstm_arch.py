"""
Model architecture
"""

import torch.nn as nn

# Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128], dropout=0.2):
        super().__init__()
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
 
        self.lstm_layers.append(nn.LSTM(input_size=input_size, hidden_size=hidden_sizes[0], batch_first=True, bidirectional=True))
        self.dropout_layers.append(nn.Dropout(dropout))
 
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(
                nn.LSTM(input_size=hidden_sizes[i-1] * 2, hidden_size=hidden_sizes[i], batch_first=True, bidirectional=True)
            )
            self.dropout_layers.append(nn.Dropout(dropout))
 
        self.fc = nn.Linear(hidden_sizes[-1] * 2, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            x, _ = lstm(x)
            x = dropout(x)
 
        last_out = x[:, -1, :]
        out = self.fc(last_out)
        return self.sigmoid(out).squeeze()
 