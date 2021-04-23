import sys

sys.path.append('model/vanilla_transformer')
sys.path.append('utils')
import my_data
from vanilla_model import *
import vanilla_config
import argparse
import torch
import time
import numpy as np
from utils.metric import *
from utils.iterator_dataset import *
import os
import sys

f = open('a.log', 'a')

def get_model(config, src_embedding_num, tag_num, embedding_matrix, embedding_dim_size, use_gpu, load_model=None):
    model = vanilla_model(config, src_embedding_num, tag_num, embedding_matrix, embedding_dim_size)(config, src_embedding_num, tag_num, embedding_matrix, embedding_dim_size)
    if load_model is not None:
        model.load_state_dict(torch.load(load_model))
    else:
        for name_, p in model.named_parameters():
            if p.dim() > 1 and str("embedding") not in name_:
                nn.init.xavier_uniform_(p)
            print(name_, p.shape, p.device)
    print('需要训练的模型参数总量:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    if True:
        model_ = model.cuda()  # 不是就地操作
    return model_


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def decode(model, my_data, use_gpu):
    # 根据输入进行解码的句子，我们输入一个句子，然后进行解码
    sentence = input("请输入您要识别的句子")
    # 接下来，我们根据词典，将句子变为 vocab 中的词语编号 转变为了tensor,下面我们进行解码
    sentence_list = list(sentence)
    sentence_list_idx = generate_char_idx([sentence_list], my_data.char_vocab)
    sentence_tensor, sentence_mask = generate_pad_idx(sentence_list_idx,my_data.char_vocab)
    # 放在gpu上面
    if use_gpu:
        sentence_tensor.cuda()
        sentence_mask.cuda()
    # 下面，进行模型计算，我们得到了path 为列表的列表
    encoder_output = model.encoder(sentence_tensor, sentence_mask, use_gpu)
    path = model.decoder(encoder_output, sentence_mask, use_gpu)
    # 然后，我们在控制台输出结果
    # 因为我们只有一个句子，所以我们取第一个
    path = path[0]
    for i, tag_idx in enumerate(path):
        print(sentence_list[i], my_data.char_vocab.itos[tag_idx], end='\t')


def eval_model(model, my_data, mode='dev', use_gpu=True):
    """
        在控制台输出 句子总量 以及处理的总时间等
       用来评估模型，得到一系列的 值，例如 precision recall f acc 值
    :param model: 模型
    :param my_data: 存放数据集的东西
    :param mode: 是dev 还是test,我们要评估的数据集
    :param use_gpu: 是否使用gpu做训练 yeap
    :return: acc precision recall f
    """
    # 根据当前 batch 来进行累加，
    # 当前累加需要累加的一些对象
    right_num = 0  # 边界和种类都对的数量累加
    predict_num = 0  # 预测出的总共的实体数量
    golden_num = 0  # 真实值中实体的数量
    char_num = 0  # 总共的字符数
    acc_char_num = 0  # 标签匹配的字符数

    data_iter = None
    model.eval()
    if mode == 'dev':
        data_iter = my_data.dev_iter
    elif mode == 'test':
        data_iter = my_data.test_iter
    else:
        print('evaluation mode choose from test and dev')
    while True:
        try:
            sentence_tensor, tag_tensor, sentence_mask = data_iter.next()
            # 转换到gpu上面
            if use_gpu:
                sentence_tensor.cuda()
                tag_tensor.cuda()
                sentence_mask.cuda()
            encoder_output = model.encoder(sentence_tensor, sentence_mask, use_gpu)
            # decode_result 是一个 列表的列表，里面存放着当前预测的词汇编号
            decode_result = model.decoder(encoder_output, sentence_mask, use_gpu)

            # 是一个累加的问题 decode_result, tag_tensor为真实值
            # 我们先不写对于单个实体种类的计算问题，只涉及全部的种类。
            # 我们根据当前的 batch 进行累加
            cur_acc_num, cur_char_num, cur_right_num, cur_golden_num, cur_predict_num = \
                get_ner_fmeasure(decode_result, tag_tensor, my_data.tag_vocab)
            right_num += cur_right_num
            predict_num += cur_predict_num
            golden_num += cur_golden_num
            char_num += cur_char_num
            acc_char_num += cur_acc_num
        except StopIteration:
            data_iter.reset_iter()
            break
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)
    accuracy = (acc_char_num+0.0)/char_num
    return accuracy, precision, recall, f_measure


def train_model(model, config, args, my_data, use_gpu):
    """
    训练模型 专用
    :param config: 模型的超参数设置
    :param args: 输入的一些保存文件的路径的一些东西
    :param my_data: 里面存放着数据集 vocab、词向量等
    :param use_gpu: 是否使用 GPU
    :return: 不知道返回什么呢哈哈哈啊哈
    """
    best_dev = -1.0  # 最好的F值是什么
    print_interval = 40 # 每十个batch输出一次
    # 准备优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.regularization)
    # 实现的一些功能：lr decay / 模型保存
    for epoch_idx in range(config.epoch_num):
        print('training epoch:', epoch_idx + 1)
        epoch_start = time.time()
        sample_start = time.time()
        epoch_loss = 0.0
        sample_loss = 0.0
        batch_id = 0
        # 把模型调整到训练时候
        model.train()
        model.zero_grad()
        # 进行学习率衰减
        lr_decay(optimizer, epoch_idx, config.lr_decay, config.lr)
        # 然后我们对epoch内的数据进行随机打乱,不知道行不行
        data_num = len(my_data.train_iter.data_list)
        print("训练数据总量：", data_num)
        # np.random.shuffle(my_data.train_iter.data_list)
        # 下面，如何取得迭代数据，并把其搞到gpu上面
        while True:
            try:
                # 不知道这样子行不行哈哈哈哈哈哈 我真是佛了
                sentence_tensor, tag_tensor, sentence_mask = my_data.train_iter.next()
                # print(sentence_tensor)
                # print(tag_tensor)
                # print(sentence_mask)
                # 转换到gpu上面
                if use_gpu:
                    sentence_tensor = sentence_tensor.cuda()
                    tag_tensor = tag_tensor.cuda()
                    sentence_mask = sentence_mask.cuda()
                batch_loss_tensor = model(sentence_tensor, tag_tensor, sentence_mask, use_gpu)
                epoch_loss += torch.sum(batch_loss_tensor, dim=-1)
                sample_loss += torch.sum(batch_loss_tensor, dim=-1)
                batch_loss = torch.mean(batch_loss_tensor, dim=-1)
                model.zero_grad()
                batch_loss.backward()  # 不知道可不可以
                optimizer.step()
                batch_id += 1
                # print('epoch:{}, batch:{}, loss:{:.4f}'.format(epoch_idx, batch_id, batch_loss.item()))
                if batch_id % print_interval == 0:
                    sample_cost = time.time() - sample_start
                    print('sample loss: epoch:{}, batch:{}, time:{:.2f}s, batch_loss:{:.4f}'.format(epoch_idx, batch_id, sample_cost, sample_loss.item()/print_interval),file=f,flush=True)
                    sample_start = time.time()
                    sample_loss = 0.0
            except StopIteration:
                my_data.train_iter.reset_iter()
                break
                
        epoch_end = time.time()
        epoch_cost = epoch_end - epoch_start
        print('Epoch:{} training finished. Time: {:.2f}s, '
              'speed: {:.2f}s/instance, total loss:{:.4f}'.format(epoch_idx + 1,epoch_cost,epoch_cost/data_num,epoch_loss.item()),file=f,flush=True)
        # acc, p, r, f = eval_model(model, my_data, use_gpu=use_gpu)
        # dev_cost = time.time() - epoch_end
        # print('dev_result: epoch:{}, time:{:.2f}, accuracy:{:.4f}, '
        #       'precision:{:.4f}, recall:{:.4f}, f_measure:{:.4f} '.format(epoch_idx+1, dev_cost, acc, p, r, f))
        # if f > best_dev:
        #     print("epoch:{} dev set exceed best f score{:4f}".format(epoch_idx+1, best_dev))
        #     best_dev = f
        #     # 如果要运行多个模型，这个就需要注意了哦！
        #     # save_model 应该是一个包含了模型名称的路径
        #     model_name = args.save_model + os.sep + 'epoch_' + str(epoch_idx)+ '.model'
        #     torch.save(model.state_dict(), model_name)


if __name__ == '__main__':
    opt = argparse.ArgumentParser(description='tuning and setting the various Transformer')
    opt.add_argument('--status', choices=['train', 'test', 'decode'], default='train')
    opt.add_argument('--save_model', default='model/saved_model/vanilla_transformer')
    opt.add_argument('--load_model', help='training from scratch or load path', default=None)
    opt.add_argument('--save_data', help='save vocab or vector', default=None)
    opt.add_argument('--load_data', help='load vocab or vector', default=None)
    opt.add_argument('--train', help='the path of train file', default='data/MSRA/msra_train_bio.txt')
    opt.add_argument('--dev', help='the path of dev file', default='data/MSRA/msra_dev_bio.txt')
    opt.add_argument('--test', help='the path of test file', default='data/MSRA/msra_test_bio.txt')
    args = opt.parse_args()
    # gpu 是否可以使用
    use_gpu = torch.cuda.is_available()
    print("gpu是否空闲：", use_gpu)
    # 将status变为小写
    status = args.status.lower()
    # 刷新缓冲区
    sys.stdout.flush()
    # 下面 我们初始化超参数以及输入
    config = vanilla_config.config()
    data = my_data.Data(args)
    # 经过如下的build_data,我们已经初始话了字词向量
    data.build_data(config.batch_size)
    # 对于下面的几句话，不管是训练还是测试，都是一样的。
    src_embedding_num = data.char_vocab.get_length()
    tag_num = data.tag_vocab.get_length() - 1
    model = get_model(config, src_embedding_num, tag_num, data.char_embedding, data.char_embedding_dim, use_gpu, args.load_model)
    # 下面 我们进行判断，进行模型训练
    if status == 'train':
        train_model(model, config, args, data, use_gpu)
    elif status == 'test':
        eval_model(model, data, mode='test', use_gpu=use_gpu)
    elif status == 'decode':
        # 输入一个句子，我们进行decode,所以当我们decode的时候，在命令行一定要load_model
        decode(model, data, use_gpu)
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")
