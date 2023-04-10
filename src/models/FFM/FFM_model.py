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


# FFM모델을 구현합니다.
# feature간의 상호작용을 파악하기 위해서 잠재백터를 두는 과정을 보여줍니다.
# FFM은 FM과 다르게 필드별로 여러개의 잠재백터를 가지므로 필드 개수만큼의 embedding parameter를 선언합니다.
class FieldAwareFactorizationMachine(nn.Module):
    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets, dtype= np.int32).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix


# 최종적인 FFM모델입니다.
# 각 필드별로 곱해져 계산된 embedding 결과를 합하고, 마지막으로 embedding 결과를 합하여 마무리합니다.
class FieldAwareFactorizationMachineModel(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.linear = FeaturesLinear(self.field_dims)
        self.ffm = FieldAwareFactorizationMachine(self.field_dims, args.embed_dim)


    def forward(self, x: torch.Tensor):
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        # return torch.sigmoid(x.squeeze(1))
        return x.squeeze(1)
