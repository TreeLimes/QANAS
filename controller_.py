import cnn_
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim
import torch


class controller(nn.Module):
    def __init__(self, num_classes, hidden_size,
                 num_layers, num_arc_params_per_layer):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_arc_params_per_layer = num_arc_params_per_layer
        self.controller = nn.LSTM(input_size=num_classes, hidden_size = hidden_size, num_layers=num_layers)

        self.classifiers = []

        for i in range(num_arc_params_per_layer):
            self.classifiers.append(nn.Sequential(
                nn.Linear(hidden_size, num_classes),
                nn.Softmax()
            ))

        self.linears = nn.ModuleList(self.classifiers)

        self.optim = optim.Adam(self.parameters(), lr=0.0006)

        self.hn = torch.randn(self.num_layers, 1, self.hidden_size)
        self.cn = torch.randn(self.num_layers, 1, self.hidden_size)
        self.x = torch.randn(1, 1, self.num_classes)

        self.is_start_state = True
        self.actions_generated = 0
        self.generate_n_actions = num_arc_params_per_layer*num_layers

    def step(self):
        if self.is_start_state:
            self.x = torch.randn(1, 1, self.num_classes)
            self.is_start_state = False

        c_ind = self.actions_generated % self.num_arc_params_per_layer
        classifier = self.classifiers[c_ind]
        output, (self.hn, self.cn) = self.controller(self.x, (self.hn, self.cn))
        self.x = classifier(output)
        m = Categorical(self.x)
        action = m.sample()

        self.actions_generated += 1
        if self.actions_generated == self.generate_n_actions:
            self.actions_generated = 0
            self.is_start_state = True
        return action, -m.log_prob(action), self.hn, self, cn

