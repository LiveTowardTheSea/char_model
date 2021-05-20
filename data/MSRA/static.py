path = '/home/wqt/char_model/data/Ontonotes_5.0/ontonotes_test_bioes'
import codecs
org = 0
per = 0
loc = 0
gpe = 0
total = 0
sentence = 0
char = 0
with codecs.open(path,'r','utf-8') as fin:
    #with codecs.open(path.rstrip('bioes')+'bio','w','utf-8') as fout:
    for line in fin.readlines():
        if line not in ['\n', '\r\n']:
            word_label = line.strip().split()
            if len(word_label) >= 2:
                word = word_label[0]
                char += 1
                label = word_label[1]
                if label[2:] == "GPE" and (label.startswith("S") or label.startswith("B")):
                    gpe += 1
                elif label[2:] == 'PERSON' and (label.startswith("S") or label.startswith("B")) :
                    per += 1
                elif label[2:] == 'LOC' and (label.startswith("S") or label.startswith("B")) :
                    loc += 1
                elif label[2:] == 'ORG' and (label.startswith("S") or label.startswith("B")) :
                    org += 1
                # if label.startswith("E"):
                #     label = 'I-'+ label[2:]
                # if label.startswith("S"):
                #     label = 'B-' +label[2:]
                # fout.write('\t'.join([word,label]))
                # fout.write('\n')
        elif line in ['\n', '\r\n']:
            sentence += 1
            # fout.write('\n')
    total = per + org + loc + gpe
    print('total',total)
    print('org',org)
    print('per',per)
    print('loc',loc)
    print('gpe',gpe)
    print('char',char)
    print('sentence',sentence)