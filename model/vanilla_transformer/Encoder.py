import torch.nn as nn
import torch
from SubLayer import *
import torch.nn.functional as F
import copy
from Pos_Encoding import *


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.context_encoder = Multi_Head_Attention(config)
        self.att_layer_norm = LayerNorm(config)
        self.position_layer = Position_Wise_Network(config)
        self.pos_layer_norm = LayerNorm(config)

    def forward(self, src, mask, use_gpu):
        # (batch_size, seq_len, d_model)
        temp, attention = self.context_encoder(src, src, mask, use_gpu)
        src = temp + src
        src = self.att_layer_norm(src)
        temp = self.position_layer(src)
        src = temp + src
        src = self.pos_layer_norm(src)
        return src, attention


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, config, src_embedding_num, embedding_matrix, embedding_dim_size):
        # embedding_num 源端单词的总数量
        super(Encoder, self).__init__()
        self.layer_num = config.layer_num
        self.embedding = nn.Embedding(src_embedding_num, embedding_dim_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        # 模型需要的embedding比较大，我们这里做一个转换
        self.trans_matrix = nn.Linear(embedding_dim_size, config.d_model)
        self.position_encoding = Position_Encoding(config)
        self.encoder_layer = get_clones(EncoderLayer(config), self.layer_num)

    def forward(self, src, mask=None, use_gpu=True):
        src_ = self.embedding(src)  # (batch,seq_len,embedding_dim_size)
        src_ = self.trans_matrix(src_)
        src_ = self.position_encoding(src_, use_gpu)
        # 这里，不知道是不是需要一个 attention 列表呢
        for i in range(self.layer_num):
            src_, attention = self.encoder_layer[i](src_, mask, use_gpu)
        return src_
