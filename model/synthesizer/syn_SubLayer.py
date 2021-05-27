import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from Dense_Attention import *
from FR_Attention import *


class Self_Attention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.head = config.head
        self.use_fr = config.use_fr
        self.d_model = config.d_model
        self.multi_v = nn.Linear(config.d_model,config.d_model)
        self.dense_atten = Dense_Attention(config.d_model,config.head,config.k_dim,config.max_len)
        if config.use_fr:
            self.fr_atten = FR_Attention(config.head,config.max_len,config.scale)
            self.rate = nn.Parameter(torch.randn(2))
        self.dropout = nn.Dropout(config.p_drop)

    def forward(self,src,mask,use_gpu):
        # src (batch,seq_len,mask)
        batch_size = src.shape[0]
        seq_len = src.shape[1]
        value = self.multi_v(src)
        value = value.view(batch_size,seq_len,self.head,-1)
        value = value.permute(0,2,1,3).contiguous() #(batch,head,seq_len,k_dim)
        dense = self.dense_atten(src) #(batch,head,seq_len,seq_len)
        if self.use_fr:
            fr = self.fr_atten(src) #(1,head,seq_len,seq_len)
            rate = F.softmax(self.rate,dim=-1)
            fr_ = rate[1] * fr
            dense_ = rate[0]* dense
            dense = dense_ + fr_
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(2)
            dense = torch.masked_fill(dense,mask=mask,value=-1e10)
        atten_weight = F.softmax(dense,dim=-1)
        atten_value = torch.matmul(atten_weight,value)
        atten_value = atten_value.permute(0,2,1,3).contiguous()
        result = atten_value.view(batch_size,seq_len,self.d_model)
        result = self.dropout(result)
        return result


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




