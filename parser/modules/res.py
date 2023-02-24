import torch.nn as nn


# class ResLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, n_layers=2, activation='relu'):
#         super(ResLayer, self).__init__()
#         if activation == 'relu':
#             activation = nn.ReLU
#         elif activation == 'tanh':
#             activation = nn.Tanh

#         self.layers = [[nn.Linear(in_dim, out_dim), activation()]]
#         if n_layers > 1:
#             for _ in range(n_layers - 1):
#                 self.layers.append(
#                     [nn.Linear(out_dim, out_dim), activation()]
#                 )
#         self.layers = [y for x in self.layers for y in x]

#         self.layers = nn.Sequential(*self.layers)

#     def forward(self, x):
#         return self.layers(x) + x

class ResLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.linear(x) + x

class ResLayerBN(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x) + x)

class ResLayerNorm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResLayerNorm, self).__init__()
        self.linear = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_dim, out_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(out_dim, out_dim)),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.linear(x) + x

if __name__ == "__main__":
    import torch
    x = torch.randn(2, 4)

    net = ResLayer(4, 4, 3, activation='tanh')
    print(net(x))