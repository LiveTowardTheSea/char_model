from numpy.lib.npyio import NpzFile
import torch
import codecs
# 迭代器功能：根据dataset、vocab 每次生成 （batch_size,seq_len)的torch数据，形成对于数据集的便利
import numpy as np
"""
功能：
    1.遍历整个数据集，形成返回下一个的迭代器
    2.返回  (batch_size,seq_len)  的 vocab_idx 及 tag_idx
        2.1 序列长度不一样，需要补全
        2.2 需要将 词语和标签 变成编号
"""


def generate_char_idx(batch_data, vocab):
    """
    这个是针对于 char 来说的，因为是只提取了单个的token
    :param batch_data: series类型的数据,有两列，要么是 sentence，要么是 label
    :param vocab: sentence 或者是label 的 vocab
    :return: idx 列表
    """
    idx = []
    for sentence in batch_data:
        each_idx = []
        for token in sentence:
            if token in vocab.vocab_list:
                each_idx.append(vocab.stoi[token])
            else:
                each_idx.append(vocab.unk_idx)
        idx.append(each_idx)
    return idx


def generate_pad_idx(idx_list, vocab):
    """
    经过上一个函数处理之后，我们可以把 token变为一系列的 idx,形成一个包含列表的列表，现在我们要做的就是
    把这个列表的列表变成tensor,并且补齐。
    :param idx_list:
    :param vocab: 进行补齐的vocab,里面含有vocab.pad
    :return: 返回 torch.tensor 以及 mask,根据pad
    """
    max_len = max([len(each_idx_list) for each_idx_list in idx_list])
    result = torch.zeros((len(idx_list), max_len), dtype=torch.long)
    for i, sentence in enumerate(idx_list):
        result[i] = torch.tensor(sentence + [vocab.pad_idx]*(max_len - len(sentence)), dtype=torch.long)
    mask = (result == vocab.pad_idx)
    return result, mask

    
def read_data(data_path):
    data_list = []
    with codecs.open(data_path, 'r', 'utf-8') as f:
        sentence_list = []
        tag_list = []
        for line in f.readlines():
            if line not in ['\n', '\r\n']:
                word_label = line.strip().split()
                if len(word_label) >= 2:
                    sentence_list.append(word_label[0])
                    tag_list.append(word_label[1])
            else:
                if len(sentence_list)>0 and len(tag_list)>0 and len(sentence_list)==len(tag_list):
                    data_list.append((sentence_list, tag_list))
                sentence_list = []
                tag_list = []
    return data_list


class data_iterator:
    def __init__(self, data_path, char_vocab, tag_vocab, batch_size):
        print("init iterator:", data_path)
        self.data_list = read_data(data_path)
        self.char_vocab = char_vocab
        self.tag_vocab = tag_vocab
        self.batch_size = batch_size
        self.offset = 0
        self.max_num = len(self.data_list) // batch_size + 1 if len(self.data_list) % batch_size != 0 else len(self.data_list) // batch_size
        # 调用桶排序算法
        self.appro_bucket_sort()
        # 接下来，根据batch_size,将模型数据打乱，使长度相近的凑在一个batch里，不同batch间长度不同。
        self.between_bucket_unsort()

         
    # 对于data_list,我们对其进行大致的桶排序，使其长度接近的大致靠在一起，减少pad,减少处理时间
    def appro_bucket_sort(self):
        """
        根据元素长度进行大致的桶排序
        :param sent_tag_list 数据列表，每一个元素为包含句子和标签的元组
        """
        # 首先，将每一个句子的长度放入列表中
        gap_len = 32
        length_list = []
        for sent,tag in self.data_list:
            length_list.append(len(sent))
        # 接下来，拿到当前句子长度的最大值和最小值。
        max_bucket_num = max(length_list) // gap_len
        min_bucket_num = min(length_list) // gap_len
        # 总共有多少个桶
        total_bucket = max_bucket_num - min_bucket_num + 1
        # 每一个桶中存放句子编号
        sentence_2_bucket = [[] for i in range(total_bucket)]
        for i,sent_len in enumerate(length_list):
            # 存放在哪个桶中,获得列表编号
            bucket_idx = sent_len // gap_len - min_bucket_num
            sentence_2_bucket[bucket_idx].append(i)
        # 首先，拉平sentence_2_idx
        sentence_2_bucket = [sent_idx for bucket in sentence_2_bucket for sent_idx in bucket]
        # 按照 sentence_2_bucket 的顺序，我们重新组织一下sent_tag_list
        orig_2_bucket= [sentence_2_bucket.index(orig_idx) for orig_idx in range(len(self.data_list))]
        new_sentence_tag_list = [x for _,x in sorted(zip(orig_2_bucket,self.data_list))]
        self.data_list = new_sentence_tag_list

    # 经过上面这个函数的处理，我们把句子长度相近的靠在了一起
    # 但如果这样，模型在开始训练时，使用短句子，后面就使用长句子，这显然是不行的。  
    def between_bucket_unsort(self):
        data_len = len(self.data_list)
        # 存放在第i位的是 原来数据 所处于的 batch_idx
        new_batch_idx = np.arange(0, (data_len-1)//self.batch_size + 1)
        np.random.shuffle(new_batch_idx)
        new_data_list = []
        for batch_idx in new_batch_idx:
            end_pos = min((batch_idx+1) * self.batch_size,data_len)
            new_data_list += self.data_list[batch_idx * self.batch_size: end_pos]
        self.data_list = new_data_list  

    def reset_iter(self):
        self.offset = 0
        # 重新洗一下数据
        np.random.shuffle(self.data_list)
        # 重新 bucket
        self.appro_bucket_sort()
        # 重新乱序
        self.between_bucket_unsort()


    def next(self):
        if self.offset == self.max_num:
            raise StopIteration
        next_idx = (self.offset + 1) * self.batch_size if self.offset != self.max_num - 1 else len(self.data_list)
        sentence_tag_data = self.data_list[self.offset * self.batch_size: next_idx]
        self.offset += 1
        sentence_data = []
        tag_data = []
        for sentence, tag in sentence_tag_data:
            sentence_data.append(sentence)
            tag_data.append(tag)
        sentence_idx = generate_char_idx(sentence_data, self.char_vocab)
        tag_idx = generate_char_idx(tag_data, self.tag_vocab)
        sentence_tensor, sentence_mask = generate_pad_idx(sentence_idx, self.char_vocab)
        tag_tensor, tag_mask = generate_pad_idx(tag_idx, self.tag_vocab)
        return sentence_tensor, tag_tensor, sentence_mask