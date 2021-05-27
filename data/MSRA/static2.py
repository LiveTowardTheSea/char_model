path = "/home/wqt/char_model/data/MSRA/msra_train_bioes"
path2 = "/home/wqt/char_model/data/MSRA/msra_train_bioes_2"
import codecs
total = [0]*100
with codecs.open(path,'r','utf-8') as fin:
    #with codecs.open(path2,'w','utf-8') as fout:
    sent_len = 0
    lines = fin.readlines()
    for i,line in enumerate(lines):
        if line not in ['\n', '\r\n']:
            sent_len += 1
            # word_label = line.strip().split()
            # if len(word_label) >= 2:
            #     fout.write('\t'.join(word_label))
            #     fout.write('\n')
            # if sent_len >= 240 and lines[i+1] not in ['\n', '\r\n']:
            #     fout.write("\n")
            #     sent_len = 0
        elif line in ['\n', '\r\n']:
            #fout.write('\n')
            total[sent_len//50] += 1
            sent_len = 0
    print(total)