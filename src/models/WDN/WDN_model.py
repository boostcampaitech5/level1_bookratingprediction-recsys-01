import numpy as np
import torch
import torch.nn as nn


# FM모델 등에서 활용되는 선형 결합 부분을 정의합니다.
class FeaturesLinear(nn.Module):
    def __init__(self, field_dims: np.ndarray, output_dim: int=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


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


# Wide: memorization을 담당하는 generalized linear model
# Deep: generalization을 담당하는 feed-forward neural network
# wide and deep model은 위의 wide 와 deep 을 결합하는 모델입니다.
# 데이터를 embedding 하여 MLP 으로 학습시킨 Deep 모델과 parameter에 bias를 더한 linear 모델을 합하여 최종결과를 도출합니다.
class WideAndDeepModel(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.linear = FeaturesLinear(self.field_dims)
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.embed_output_dim = len(self.field_dims) * args.embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, args.mlp_dims, args.dropout)


    def forward(self, x: torch.Tensor):
        embed_x = self.embedding(x)
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return x.squeeze(1)
