import numpy as np
import torch
import torch.nn as nn


# factorization을 통해 얻은 feature를 embedding 합니다.
class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


# cross product transformation을 구현합니다.
class CrossNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])


    def forward(self, x: torch.Tensor):
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


# NCF 모델은 MLP와 GMF를 합하여 최종 결과를 도출합니다.
# MLP을 구현합니다.
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)


    def forward(self, x):
        return self.mlp(x)


# Crossnetwork 결과를 MLP layer에 넣어 최종결과를 도출합니다.
class DeepCrossNetworkModel(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.embed_output_dim = len(self.field_dims) * args.embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, args.num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, args.mlp_dims, args.dropout, output_layer=False)
        self.cd_linear = nn.Linear(args.mlp_dims[0], 1, bias=False)


    def forward(self, x: torch.Tensor):
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        x_out = self.mlp(x_l1)
        p = self.cd_linear(x_out)
        return p.squeeze(1)
