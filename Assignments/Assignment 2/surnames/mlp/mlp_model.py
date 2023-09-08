from torch import nn
import torch.nn.functional as F

class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MultilayerPerceptron, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(last_dim, hidden_dim))
            last_dim = hidden_dim

        self.output_layer = nn.Linear(last_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x_in):
        for hidden_layer in self.hidden_layers:
            x_in = F.relu(hidden_layer(x_in))

        output = self.softmax(self.output_layer(x_in))
        return output
