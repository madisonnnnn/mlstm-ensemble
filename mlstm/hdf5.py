import csv
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np



def seq2kmer(seq, k):
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers
def load_vocab(vocab_file):
    word2id = {}
    with open(vocab_file,"r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")  # 删除 string 字符串末尾的指定字符（默认为空格）
        word2id[token] = index
    id2word = dict([val, key] for key, val in word2id.items())
    return word2id, id2word
def text2tokens(word2id, text, do_lower_case=True):
    output_tokens = []
    text = text.split()
    text_list =list(text)
    for i in text_list:
        if i in word2id.keys():
            output_tokens.append(word2id[i])
        else:
            output_tokens.append(-1)
    return output_tokens
def read_data(file):
    data = []
    lable = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            #print(row)
            #读列
            data.append(row[0])
            lable.append(row[1])
    return data,lable
def to_one_hot(labels, Label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])
    return one_hot_label
# def seq2one_hot(seq,lable1,kmer,num_classes):
#     feature=[]
#     target=[]
#     for i in range(len(seq)):
#         record1 = seq2kmer(seq[i],kmer)
#         # record1 = np.array(record1)
#         word2id, id2word = load_vocab('./vocab1mer.txt')
#         b = np.array(text2tokens(word2id, record1))
#         #b = torch.from_numpy(b).to(torch.int64)
#         feature1 = to_one_hot(b, num_classes)
#         classes=lable1[i]
#         feature.append(feature1)
#         target.append(list(classes))
#     return feature, target

def seq2one_hot(seq,lable1,kmer,num_classes):

    record1 = seq2kmer(seq,kmer)
    # record1 = np.array(record1)
    word2id, id2word = load_vocab('./vocab5mer.txt')
    b = np.array(text2tokens(word2id, record1))
    #b = torch.from_numpy(b).to(torch.int64)
    feature1 = to_one_hot(b, num_classes)
    classes=lable1
    # feature.append(feature1)
    # target.append(list(classes))
    return feature1, list(classes)



class HDF5DatasetWriter:
    def __init__(self, dims1, dims2, outputPath,long):
        # 构建两种数据，一种用来存储图像特征一种用来存储标签
        self.db = h5py.File(outputPath, "w")
        self.images = self.db.create_dataset("images", dims1,  chunks=(1, long, 1025),compression='gzip',compression_opts=5)
        self.labels = self.db.create_dataset("labels", dims2,  chunks=(1, 1))
        # 用来进行计数
        self.idx = 0

    def add(self, images, labels):
        print(self.idx)
        self.images[self.idx, ...] = images
        self.labels[self.idx, ...] = labels
        self.idx += 1

    def close(self):
        self.db.close()

#开始创建h5文件并写入数据
def create_h5(h5_path,csv_path,long):
    num = 41658
    #num = 154275
    #num = 258258
    writer = HDF5DatasetWriter((num, long,1025), (num, 1), h5_path,long)
    for line in open(csv_path,'r'):
        data = line[0:-3]
        label =line[-2]
        X, Y = seq2one_hot(data, label, 5, 1025)
        
        writer.add(X, Y)
    writer.close()

class RNADataset(Dataset):
    def __init__(self,path, bsize = 1):
        self.path = path
        self.bsize = bsize
        f = h5py.File(self.path,'r')
        self.data_img = f["images"]
        self.data_cls = f["labels"]
        self.bcount = int( (self.data_img.shape[0] - 0.1) // self.bsize + 1 )
    def __len__(self):
        return self.bcount
    def __getitem__(self, idx):
        data_x = np.array(self.data_img[idx*self.bsize:(idx+1)*self.bsize,...])
        input = torch.from_numpy(data_x).to(torch.float32).squeeze(0)
        data_y = np.array(self.data_cls[idx*self.bsize:(idx+1)*self.bsize,...])
        a=self.data_cls[idx*self.bsize:(idx+1)*self.bsize,...]
        # print(a)
        targets = torch.from_numpy(data_y).to(torch.float32).long().squeeze(0)
        return input, targets

# dataset =RNADataset('./train1.h5')
#
# trainloader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=8)
#
# for i, data in enumerate(trainloader, 0):
#     # Get inputs
#     inputs, targets = data
#     targets=targets.squeeze(1)
#     print(targets)


if __name__ == '__main__':
    
    lenth=[41,101,201,301,401,601,801,1001]
    trainset = ['train1', 'train2', 'train3', 'train4', 'train5', 'train6', 'train7','train8', 'train9', 'train10']
    for i in range(8):
        for a in range(10):
            h5 ='./全转录组_h5/HEK293_abacm/'+str(lenth[i])+'/5mer/'+trainset[a]+'.h5'
            csv= './全转录组/HEK293_abacm/'+str(lenth[i])+'/p/'+trainset[a]+'.csv'
            create_h5(h5,csv,lenth[i]-4)
    # rng = np.random.RandomState(123)
    # f = h5py.File('./全转录组_h5/A549/1001/1mer/train1bi10.h5', 'r')
    # data_img =f['images']
    # data_cls =f["labels"]
    # a = data_img[0:]
    # b = data_cls[0:].squeeze()
    # print(a.shape)
    # kf = KFold(n_splits=5, shuffle=True, random_state=rng)
    # for train_index, test_index in kf.split(a):
    #     print(a[train_index].shape)
    #     data = a[train_index].reshape(22440,5005)
    #     print(data.shape)
    #     print('****************')



