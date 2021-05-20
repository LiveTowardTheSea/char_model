
class vanilla_config:
    def __init__(self):
        self.model_name = 'vanilla'
        self.head = 4
        self.k_dim = 64
        self.d_model = 256
        self.p_drop = 0.1
        self.d_ff = 1024
        self.layer_num = 4
        self.lr = 0.001
        self.regularization = 0.008
        self.lr_decay = 0.05
        self.batch_size = 32
        self.epoch_num = 30
        self.momentum = 0.9

class universal_config:
    def __init__(self):
        # 模型名称是什么
        self.model_name = 'universal'
        self.max_step = 4
        self.d_model = 256
        self.d_ff = 1024
        self.k_dim = 64
        self.head = 4
        self.p_drop = 0.1
        self.threshold = 0.99
        self.batch_size = 16
        self.lr = 0.001
        self.lr_decay = 0.05
        self.regularization=0.008
        self.epoch_num=30
        self.momentum =0.9
