import torch.nn as nn
import torch.nn.functional as F
import math
import torch


class Multi_Head_Attention(nn.Module):
    def __init__(self, config):
        # 输入输出均为 d_model  v_dim = d_model / head_num
        super(Multi_Head_Attention, self).__init__()
        self.head = config.head
        self.k_dim = config.k_dim
        self.q_multi_linear = nn.Linear(config.d_model, config.head * config.k_dim)
        self.k_multi_linear = nn.Linear(config.d_model, config.head * config.k_dim)
        self.v_multi_linear = nn.Linear(config.d_model, config.d_model)
        self.o_matrix = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.p_drop)

    def forward(self, x, y, mask=None, use_gpu=True):
        # x (batch_size, seq_len, encoder_output_dim)
        """
        产生 attend 之后的上下文向量
        :param x: query
        :param y: key value
        :param mask: 采用了广播，形状应该为 (batch_size,seq_len)
        :return: 上下文向量：(batch_size,seq_len,d_model),以及 attention (batch,head,q_seq_len,v_seq_len)
        """
        query = self.q_multi_linear(x)
        key = self.k_multi_linear(y)  # (batch_size, seq_len, head_num * k_dim)
        value = self.v_multi_linear(y)
        query = query.view(query.shape[0], query.shape[1], self.head, -1)
        key = key.view(key.shape[0], key.shape[1], self.head, -1)
        value = value.view(value.shape[0], value.shape[1], self.head, -1)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        qk_result = query.matmul(key)
        qk_result = qk_result / math.sqrt(self.k_dim)  # (batch_Size, head_num, q_seq_len, k_seq_len)
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(2)
            # print('mask shape:', mask.shape)
            qk_result = qk_result.masked_fill(mask, -1e10)
        qk_result = F.softmax(qk_result, dim=3)  # (batch_size,head_num,q_seq_len,k_seq_len)
        qkv_result = qk_result.matmul(value)  # (batch_size,head_num,q_seq_len,v_dim)
        qkv_result = qkv_result.permute(0, 2, 1, 3)
        qkv_result = qkv_result.contiguous().view(qkv_result.shape[0], qkv_result.shape[1], -1)
        qkv_result = self.o_matrix(qkv_result)
        qkv_result = self.dropout(qkv_result)
        return qkv_result, qk_result
        #return qkv_result


class Position_Wise_Network(nn.Module):
    # FCNN encoder的第二层
    def __init__(self, config):
        super(Position_Wise_Network, self).__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.p_drop)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    # 在预测是也要使用的哦
    def __init__(self, config, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.size = config.d_model
        # 突然想起来这么初始化会不会不太对
        self.gamma = nn.Parameter(torch.randn(self.size))
        self.beta = nn.Parameter(torch.randn(self.size))
        self.eps = eps

    def forward(self, x):
        # x (batch_size,seq_len,d_model)
        x = (x - torch.mean(x, dim=-1, keepdim=True))/(torch.std(x, dim=-1, keepdim=True)+ self.eps)
        norm = x * self.gamma + self.beta
        return norm




