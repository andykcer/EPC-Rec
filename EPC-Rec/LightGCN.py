import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, n_layers, keep_prob, dropout, graph, device):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.keep_prob = keep_prob
        self.dropout = dropout
        self.device = device

        # 初始化图结构并转移到设备
        self.graph = graph.to(device)

        # 定义嵌入层
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items , latent_dim)

        # 初始化权重
        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.user_embedding.weight, std=0.001)
        nn.init.normal_(self.item_embedding.weight, std=0.001)

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = (torch.rand(len(values), device=self.device) + keep_prob).int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        return torch.sparse.FloatTensor(index.t(), values, size).to(self.device)

    def __dropout(self):
        if self.training and self.keep_prob < 1:
            return self.__dropout_x(self.graph, self.keep_prob)
        else:
            return self.graph

    def computer(self):
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        g_droped = self.__dropout() if self.dropout else self.graph

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items =torch.split(light_out, [self.num_users, self.num_items ])# torch.Size([58907, 100]) torch.Size([2584, 100])
        #items = items[1:]
        #print(users.shape, items.shape)

        return users, items
