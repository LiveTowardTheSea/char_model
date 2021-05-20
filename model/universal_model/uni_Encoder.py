
import torch
import torch.nn as nn
from uni_Encoder_Layer import *
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, config, src_embedding_num, embedding_dim_size, embedding_matrix):
        super(Encoder, self).__init__()
        self.config = config
        # 定义input embedding system
        self.embed = nn.Embedding(src_embedding_num, embedding_dim_size)
        self.embed.weight.data.copy_(torch.from_numpy(embedding_matrix))
        #因为转移函数要共享参数，所以我们这里直接定义一些列参数，然后传递到每一层里去。
        self.first_weight_param = nn.Parameter(torch.randn(config.d_ff, config.d_model))
        self.first_bias_param = nn.Parameter(torch.randn(config.d_ff))
        self.second_weight_param = nn.Parameter(torch.randn(config.d_model, config.d_ff))
        self.second_bias_param = nn.Parameter(torch.randn(config.d_model))
        #下面，我们定义每一个layer, 传过去当前在第几层，并且传递我们需要共享的参数。
        self.encoder_layer = nn.ModuleList([Encoder_Layer(config, i, self.first_weight_param, self.first_bias_param,
                                                          self.second_weight_param, self.second_bias_param) for i in
                                            range(config.max_step)])
        #self.encoder_layer = nn.ModuleList([Encoder_Layer(config, i) for i in range(config.max_step)])
        # 用于决定当前token是否在该步骤停止计算
        self.trans_halting_prob = nn.Linear(config.d_model, 1)

    def continue_loop_condition(self, halting_prob_):
        if torch.lt(halting_prob_, self.config.threshold).float().sum() > 0.0:
            return True
        else:
            return False


    def forward(self, src, mask=None, use_gpu=True, return_attn=False):
        # src(batch_size,seq_len)
        src_ = self.embed(src)
        # (batch_size,seq_len,d_model)
        # 当前的状态 经过第一个layer之后
        state = src_
        # 之前的状态，我们累加的一个工具
        previous_state = torch.zeros(state.size())
        # 用于进行概率累加
        halting_prob = torch.zeros(state.size()[0], state.size()[1])
        # 一个位置暂停之后我们设置其remainder
        remainder = torch.zeros(state.size()[0], state.size()[1])
        # 累加每一个位置更新个数
        updates_num = torch.zeros(state.size()[0], state.size()[1])
        # 加载到 gpu 上面
        # 是否需要返回每一层的注意力矩阵
        atten_list = []
        if use_gpu:
            previous_state = previous_state.cuda()
            halting_prob = halting_prob.cuda()
            remainder = remainder.cuda()
            updates_num = updates_num.cuda()
        # 如果还没有遍历完所有的层，对于所有batch,所有token,就已经停止了，
        # 那我们就不再需要一直到所有的层都遍历结束了，我们可以提前结束循环，拿到当前层的结果
        for i in range(self.config.max_step):
            # 如果所有的都已经停止，我们也没有计算的必要了：
            if not self.continue_loop_condition(halting_prob):
                break
            # 当intermediate prob  停止概率 是什么
            p = self.trans_halting_prob(state)
            p = F.sigmoid(p).squeeze(-1)  # (batch_size,seq_len)
            # 首先 我们获得 当前还没有停止的所有batch 及其 symbol pos
            still_running = torch.lt(halting_prob, 1.0).float()
            # 累加当前步数的intermediate prob之后，有可能会停止
            # 当前层停止：
            new_halt = torch.ge(halting_prob + p * still_running, self.config.threshold).float() * still_running
            # 经过当前层 依然不停止
            still_running = torch.lt(halting_prob + p * still_running, self.config.threshold).float() * still_running
            # 然后，我们更新halting_prob
            # 对于在当前层，依然不停止的，我们只需要累加进去概率
            halting_prob += still_running * p
            remainder += new_halt * (1 - halting_prob)
            # 对于当前层要停止的，我们将他的 halting_prob 变为 1
            halting_prob += new_halt * remainder
            # 累加所有symbol计算步数
            updates_num += still_running + new_halt
            # 进行累加，计算当前状态的累加：
            # new_halt  权重为 remainder
            # 早就停止的，权重为 0
            # 经过当前步骤仍然不停止的，权重为 p
            #state, attention = self.encoder_layer[i](state, mask, use_gpu)
            state = self.encoder_layer[i](state, mask, use_gpu)
            # if return_attn:
            #     atten_list.append(attention.cpu().numpy())
            update_weights = (new_halt * remainder + still_running * p).unsqueeze(-1)
            previous_state = state * update_weights + previous_state
        return previous_state, atten_list, updates_num.detach().cpu().numpy()