# 主要用于一些超参数：模型参数的配置
class config:
    def __init__(self):
        self.head = 4
        self.k_dim = 64
        self.d_model = 256
        self.p_drop = 0.001
        self.d_ff = 512
        self.layer_num = 4
        self.lr = 0.001
        self.regularization = 0.1
        self.lr_decay = 0.05
        self.batch_size = 48
        self.epoch_num = 10



