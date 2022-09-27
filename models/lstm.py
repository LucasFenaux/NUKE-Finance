import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LSTM, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.lstm = nn.LSTM(input_size=num_inputs, hidden_size=16, batch_first=True)
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x, hidden=None):
        # x [batch_size, sequence_length, num_inputs]
        out, hidden = self.lstm(x, hidden)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out, hidden