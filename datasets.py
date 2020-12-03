import pandas as pd
import numpy as np
import torch
from torch.utils import data  # 获取迭代数据

def label2num(label):
    if label == 'neutral':
        return 0
    if label == 'positive':
        return 1
    if label == 'negative':
        return 2

def MVSA_single_text_data_load(model_name):


    # 读取label数据
    label_path = '../Preprocessed data/singleLabel_new.txt'
    ID = []
    text_label = []
    image_label = []
    all_label = []
    f = open(label_path, 'r')
    for line in f:
        ID.append(int(line.split()[0]))
        text_label.append(label2num(line.split()[1].split(',')[0]))
        image_label.append(label2num(line.split()[1].split(',')[1]))
        all_label.append(label2num(line.split()[1].split(',')[2]))
    f.close()
    text_feature = []

    #text_feature_path = '../Preprocessed data/raw_txt_XLNet/single/'  ##文本向量
    text_feature_path = '../Preprocessed data/text-bert-single/'  ##文本向量

    for i in range(1, 5200):
        if i not in ID:
            continue
        filename = text_feature_path + str(i) + '.txt'
        try:
            with open(filename, 'r') as f:
                line = f.readline()
                line = line.strip()
                line = eval(line)[0]
                if line == 0:
                    line = [0 for _ in range(1024)]
                text_feature.append(list(line))
                # print(lis)
                f.close()
        except:
            continue
    print("文本数据读取完成。")

    dataset_len = len(text_feature)

    train_text_data = torch.from_numpy(np.array(text_feature[:int(dataset_len * 9 / 10)]))
    train_label_data = torch.from_numpy(np.array(all_label[:int(dataset_len * 9 / 10)]))
    test_text_data = torch.from_numpy(np.array(text_feature[int(dataset_len * 9 / 10):]))
    test_label_data = torch.from_numpy(np.array(all_label[int(dataset_len * 9 / 10):]))

    train_long = len(train_text_data)
    test_long = len(test_text_data)
    print("train_long: ", train_long)
    print("test_long: ", test_long)

    if model_name == 'CNN':
        train_text_data = train_text_data.resize_(int(dataset_len * 9 / 10), 1, 32, 32)
        test_text_data = test_text_data.resize_(int(dataset_len) - int(dataset_len * 9 / 10), 1, 32, 32)
    if model_name == 'RNN':
        train_text_data = train_text_data.resize_(int(dataset_len * 9 / 10), 1, 1024)
        test_text_data = test_text_data.resize_(int(dataset_len) - int(dataset_len * 9 / 10), 1, 1024)
    if model_name == 'AttnRNN':
        train_text_data = train_text_data.resize_(int(dataset_len * 9 / 10), 1, 1024)
        test_text_data = test_text_data.resize_(int(dataset_len) - int(dataset_len * 9 / 10), 1, 1024)

    """

    train_text_data = torch.from_numpy(np.array(text_feature[:int(dataset_len * 4 / 5)]))
    train_label_data = torch.from_numpy(np.array(text_label[:int(dataset_len * 4 / 5)]))
    test_text_data = torch.from_numpy(np.array(text_feature[int(dataset_len * 4 / 5):]))
    test_label_data = torch.from_numpy(np.array(text_label[int(dataset_len * 4 / 5):]))

    train_long = len(train_text_data)
    test_long = len(test_text_data)
    print("train_long: ", train_long)
    print("test_long: ", test_long)
    if model_name == 'CNN':
        train_text_data = train_text_data.resize_(train_long, 1, 32, 32)
        test_text_data = test_text_data.resize_(test_long, 1, 32, 32)
    if model_name == 'RNN':
        train_text_data = train_text_data.resize_(train_long, 1, 1024)
        test_text_data = test_text_data.resize_(test_long, 1, 1024)
    """
    train_data = zip(train_text_data, train_label_data)
    test_data = zip(test_text_data, test_label_data)

    train_loader = data.DataLoader(list(train_data), batch_size=1, shuffle=False)
    test_loader = data.DataLoader(list(test_data), batch_size=1, shuffle=False)

    return train_loader, test_loader

def MVSA_single_img_data_load(model_name):


    # 读取label数据
    label_path = '../Preprocessed data/singleLabel_new.txt'
    ID = []
    text_label = []
    image_label = []
    all_label = []
    f = open(label_path, 'r')
    for line in f:
        ID.append(int(line.split()[0]))
        text_label.append(label2num(line.split()[1].split(',')[0]))
        image_label.append(label2num(line.split()[1].split(',')[1]))
        all_label.append(label2num(line.split()[1].split(',')[2]))
    f.close()

    img_feature = []

    img_feature_path = '../Preprocessed data/raw_img_VGGNet16_Normalized/single/'  ##文本向量

    for i in range(1, 5200):
        if i not in ID:
            continue
        filename = img_feature_path + str(i) + '.txt'
        try:
            with open(filename, 'r') as f:
                line = f.readline()
                line = line.strip()
                line = eval(line)
                img_feature.append(line)
                # print(lis)
                f.close()
        except:
            continue
    print("图片数据读取完成。")

    dataset_len = len(img_feature)


    train_text_data = torch.from_numpy(np.array(img_feature[:int(dataset_len * 9 / 10)]))
    #train_label_data = torch.from_numpy(np.array(image_label[:int(dataset_len * 9 / 10)]))
    train_label_data = torch.from_numpy(np.array(all_label[:int(dataset_len * 9 / 10)]))
    test_text_data = torch.from_numpy(np.array(img_feature[int(dataset_len * 9 / 10):]))
    #test_label_data = torch.from_numpy(np.array(image_label[int(dataset_len * 9 / 10):]))
    test_label_data = torch.from_numpy(np.array(all_label[int(dataset_len * 9 / 10):]))

    train_long = len(train_text_data)
    test_long = len(test_text_data)
    print("train_long: ", train_long)
    print("test_long: ", test_long)

    if model_name == 'CNN':
        train_text_data = train_text_data.resize_(train_long, 1, 32, 32)
        test_text_data = test_text_data.resize_(test_long, 1, 32, 32)
    if model_name == 'RNN':
        train_text_data = train_text_data.resize_(train_long, 1, 1024)
        test_text_data = test_text_data.resize_(test_long, 1, 1024)
    if model_name == 'AttnRNN':
        train_text_data = train_text_data.resize_(train_long, 1, 1024)
        test_text_data = test_text_data.resize_(test_long, 1, 1024)

    """

    train_text_data = torch.from_numpy(np.array(text_feature[:int(dataset_len * 4 / 5)]))
    train_label_data = torch.from_numpy(np.array(text_label[:int(dataset_len * 4 / 5)]))
    test_text_data = torch.from_numpy(np.array(text_feature[int(dataset_len * 4 / 5):]))
    test_label_data = torch.from_numpy(np.array(text_label[int(dataset_len * 4 / 5):]))

    train_long = len(train_text_data)
    test_long = len(test_text_data)
    print("train_long: ", train_long)
    print("test_long: ", test_long)
    if model_name == 'CNN':
        train_text_data = train_text_data.resize_(train_long, 1, 32, 32)
        test_text_data = test_text_data.resize_(test_long, 1, 32, 32)
    if model_name == 'RNN':
        train_text_data = train_text_data.resize_(train_long, 1, 1024)
        test_text_data = test_text_data.resize_(test_long, 1, 1024)
    """
    train_data = zip(train_text_data, train_label_data)
    test_data = zip(test_text_data, test_label_data)


    train_loader = data.DataLoader(list(train_data), batch_size=1, shuffle=False)
    test_loader = data.DataLoader(list(test_data), batch_size=1, shuffle=False)

    print(test_loader)
    print(test_text_data)

    return train_loader, test_loader


def MVSA_single_all_label_load():


    # 读取label数据
    label_path = '../Preprocessed data/singleLabel_new.txt'
    ID = []
    text_label = []
    image_label = []
    all_label = []
    f = open(label_path, 'r')
    for line in f:
        ID.append(int(line.split()[0]))
        text_label.append(label2num(line.split()[1].split(',')[0]))
        image_label.append(label2num(line.split()[1].split(',')[1]))
        all_label.append(label2num(line.split()[1].split(',')[2]))

    print("标签数据读取完成。")


    return ID, text_label, image_label, all_label



def MVSA_multiple_all_label_load():

    def labelchange(label):
        if label == 0:
            return 1
        elif label == 1:
            return 0
        else:
            return 2



    # 读取label数据
    label_path = '../Preprocessed data/multipleLabel.txt'
    ID = []
    all_label = []
    f = open(label_path, 'r')
    for line in f:
        ID.append(int(line.split(',')[0]))
        all_label.append(labelchange(int(line.split(',')[1])))


    print("标签数据读取完成。")


    return ID, all_label


def MVSA_single_correlation_load():
    def Z_ScoreNormalization(x):
        x = (x - np.average(x)) / np.std(x)
        x = (x - np.min(x)) / (np.max(x) - np.min(x)) + 0.5
        return x

    #cor_path = '../Preprocessed data/single_correlation.txt'
    cor_path = '../Preprocessed data/single_bert_VGG_cos.txt'
    cor = []
    f = open(cor_path, 'r')
    for line in f:
        cor.append(float(line))
    cor_normal = Z_ScoreNormalization(cor)

    print("相关性读取完毕")
    return cor_normal


def MVSA_multiple_correlation_load():
    def Z_ScoreNormalization(x):
        x = (x - np.average(x)) / np.std(x)
        x = (x - np.min(x)) / (np.max(x) - np.min(x)) + 0.5
        return x

    #cor_path = '../Preprocessed data/multiple_correlation.txt'
    cor_path = '../Preprocessed data/multiple_bert_VGG_cos.txt'

    cor = []
    f = open(cor_path, 'r')
    for line in f:
        cor.append(float(line))
    cor_normal = Z_ScoreNormalization(cor)

    print("相关性读取完毕")
    return cor_normal