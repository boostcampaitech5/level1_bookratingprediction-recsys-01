import numpy as np
import torch
import torch.nn as nn
from src.models.FFM import FeaturesLinear, FieldAwareFactorizationMachine
from src.models.DCN import FeaturesEmbedding, CrossNetwork, MultiLayerPerceptron

class FieldAwareFactorizationDeepCrossNetworkModel(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        # FFM
        self.ff_field_dims = data['field_dims']
        self.ff_linear = FeaturesLinear(self.ff_field_dims)
        self.ffm = FieldAwareFactorizationMachine(self.ff_field_dims, args.ff_embed_dim)
        
        # DCN
        self.dcn_field_dims = self.ff_field_dims[:2]
        self.dcn_embedding = FeaturesEmbedding(self.dcn_field_dims, args.dcn_embed_dim)
        self.embed_output_dim = len(self.dcn_field_dims) * args.dcn_embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, args.num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, args.mlp_dims, args.dropout, output_layer=False)
        self.cd_linear = nn.Linear(args.mlp_dims[0], 1, bias=False)
        
        self.f_linear = nn.Linear(2, 1, bias=False)


    def forward(self, x: torch.Tensor):
        # FFM
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        ffm_x = self.ff_linear(x) + ffm_term
        
        # DCN
        dcn_x = x[:,:2]
        dcn_embed_x = self.dcn_embedding(dcn_x).view(-1, self.embed_output_dim)
        dcn_x_l1 = self.cn(dcn_embed_x)
        dcn_x_out = self.mlp(dcn_x_l1)
        dcn_p = self.cd_linear(dcn_x_out)
        
        p = self.f_linear(torch.cat([ffm_x, dcn_p], dim=1))
        return p.squeeze(1)