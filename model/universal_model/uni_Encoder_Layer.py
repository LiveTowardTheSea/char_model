import torch.nn as nn
from uni_Sublayer import *


class Encoder_Layer(nn.Module):
    def __init__(self, config, time_step,first_weight_param, first_bias_param,second_weight_param,second_bias_param):
        super(Encoder_Layer, self).__init__()
        self.pos_time_encode = Time_Pos_Encoding(config, time_step)
        self.attention = Multi_Head_Attention(config)
        self.atten_layer_norm = LayerNorm(config)
        self.transition_func = FCNN_transition_func(config,first_weight_param, first_bias_param,second_weight_param,second_bias_param)
        self.trans_layer_norm = LayerNorm(config)

    def forward(self, src, mask, use_gpu):
        src_ = self.pos_time_encode(src, use_gpu)
        #temp,attention_ = self.attention(src, mask, use_gpu)
        temp  = self.attention(src, mask, use_gpu)
        src_ = src_ + temp
        src_ = self.atten_layer_norm(src_)
        temp = self.transition_func(src_)
        src_ = src_ + temp
        src_ = self.trans_layer_norm(src_)
        #return src_, attention_
        return src_
