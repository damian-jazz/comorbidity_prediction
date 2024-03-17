from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim=922, output_dim=13):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.linear_stack = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, self.output_dim),       
        )

    def forward(self, x):
        x_ls = self.linear_stack(x)
        return x_ls