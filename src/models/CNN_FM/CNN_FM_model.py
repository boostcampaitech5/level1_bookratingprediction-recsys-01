import numpy as np
import torch
import torch.nn as nn


# feature 사이의 상호작용을 효율적으로 계산합니다.
class FactorizationMachine(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.v = nn.Parameter(torch.rand(input_dim, latent_dim), requires_grad = True)
        self.linear = nn.Linear(input_dim, 1, bias=True)


    def forward(self, x):
        linear = self.linear(x)
        square_of_sum = torch.mm(x, self.v) ** 2
        sum_of_square = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        output = linear + (0.5 * pair_interactions)
        return output


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


# 이미지 특징 추출을 위한 기초적인 CNN Layer를 정의합니다.
class CNN_Base(nn.Module):
    def __init__(self, ):
        super(CNN_Base, self).__init__()
        self.cnn_layer = nn.Sequential(
                                        nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2),
                                        nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2),
                                        )
    def forward(self, x):
        x = self.cnn_layer(x)
        x = x.view(-1, 12 * 1 * 1)
        return x


# 기존 유저/상품 벡터와 이미지 벡터를 결합하여 FM으로 학습하는 모델을 구현합니다.
class CNN_FM(torch.nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = np.array([len(data['user2idx']), len(data['isbn2idx'])], dtype=np.uint32)
        self.embedding = FeaturesEmbedding(self.field_dims, args.cnn_embed_dim)
        self.cnn = CNN_Base()
        self.fm = FactorizationMachine(
                                        input_dim=(args.cnn_embed_dim * 2) + (12 * 1 * 1),
                                        latent_dim=args.cnn_latent_dim,
                                        )


    def forward(self, x):
        user_isbn_vector, img_vector = x[0], x[1]
        user_isbn_feature = self.embedding(user_isbn_vector)
        img_feature = self.cnn(img_vector)
        feature_vector = torch.cat([
                                    user_isbn_feature.view(-1, user_isbn_feature.size(1) * user_isbn_feature.size(2)),
                                    img_feature
                                    ], dim=1)
        output = self.fm(feature_vector)
        return output.squeeze(1)
