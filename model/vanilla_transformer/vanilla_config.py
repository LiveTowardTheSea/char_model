# 主要用于一些超参数：模型参数的配置
class config:
    def __init__(self):
        self.head = 8
        self.k_dim = 64
        self.d_model = 512
        self.p_drop = 0.1
        self.d_ff = 2048
        self.layer_num = 6
        self.lr = 0.1
        self.regularization = 1e-8
        self.lr_decay = 0.05
        self.batch_size = 50
        self.epoch_num = 1



