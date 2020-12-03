import torch

from torch.autograd import Variable # 获取变量
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from torch.utils import data  # 获取迭代数据
from sklearn.metrics import f1_score


from datasets import MVSA_single_text_data_load, MVSA_single_img_data_load, MVSA_single_all_label_load, MVSA_single_correlation_load
from CNN import CNNnet_onlyImg
from RNN import RNNnet, AttnRNN
from co_attention import CoAtt, CoAtt_and_Independent, multi_CoAtt_and_Independent2
import co_attention
from sklearn.model_selection import train_test_split

learning_rate = 0.0003

#model = CoAtt_and_Independent()            #不用MHFF
model = multi_CoAtt_and_Independent2()     #用MHFF
#model = RNNnet()                           #只用文本
#model = CNNnet_onlyImg()                   #只用图片

def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()

#print(params_count(model))
#input()


ID, text_label, image_label, all_label = MVSA_single_all_label_load()
correlation = MVSA_single_correlation_load()

text_feature = []
img_feature = []
text_feature2 = []
img_feature2 = []

text_feature_path = '../Preprocessed data/text-bert-single/'  ##文本向量
img_feature_path = '../Preprocessed data/raw_img_VGGNet16_Normalized/single/'  ##文本向量
text_feature_path2 = '../Preprocessed data/raw_txt_XLNet/single/'  ##文本向量
img_feature_path2 = '../Preprocessed data/img_resnet50/single/'  ##文本向量

save_path = './exam/MVSA-single/消融实验/MVSA-single-cos-directly'+'-lr='+str(learning_rate)+'-DROPOUT_R='+str(co_attention.DROPOUT_R)+\
            '-HIDDEN_SIZE='+str(co_attention.HIDDEN_SIZE)+'-fus_='+str(co_attention.fus_)
import sys

class Logger(object):
    def __init__(self, filename = save_path + '.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger(stream=sys.stdout)

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
    filename = img_feature_path + str(i) + '.txt'
    try:
        with open(filename, 'r') as f:
            line = f.readline()
            line = line.strip()
            line = eval(line)
            if line == 0:
                line = [0 for _ in range(1024)]
            img_feature.append(list(line))
            # print(lis)
            f.close()
    except:
        continue


    filename = text_feature_path2 + str(i) + '.txt'
    try:
        with open(filename, 'r') as f:
            line = f.readline()
            line = line.strip()
            line = eval(line)
            if line == 0:
                line = [0 for _ in range(1024)]
            text_feature2.append(list(line))
            # print(lis)
            f.close()
    except:
        continue
    filename = img_feature_path2 + str(i) + '.txt'

    try:
        with open(filename, 'r') as f:
            line = f.readline()
            line = line.strip()
            line = eval(line)
            if line == 0:
                line = [0 for _ in range(1024)]
            img_feature2.append(list(line))
            # print(lis)
            f.close()
    except:
        continue

    if i % 1000 == 0:
        print(i, "读取完成")
print("数据读取完成。")


dataset_len = len(text_feature)

"""
train_text_data = torch.from_numpy(np.array(text_feature[:int(dataset_len * 8 / 10)]))
train_img_data = torch.from_numpy(np.array(img_feature[:int(dataset_len * 8 / 10)]))
test_text_data = torch.from_numpy(np.array(text_feature[int(dataset_len * 8 / 10):]))
test_img_data = torch.from_numpy(np.array(img_feature[int(dataset_len * 8 / 10):]))
train_text2_data = torch.from_numpy(np.array(text_feature2[:int(dataset_len * 8 / 10)]))
train_img2_data = torch.from_numpy(np.array(img_feature2[:int(dataset_len * 8 / 10)]))
test_text2_data = torch.from_numpy(np.array(text_feature2[int(dataset_len * 8 / 10):]))
test_img2_data = torch.from_numpy(np.array(img_feature2[int(dataset_len * 8 / 10):]))


train_long = len(train_text_data)
test_long = len(test_text_data)
print("train_long: ", train_long)
print("test_long: ", test_long)

train_label = torch.from_numpy(np.array(all_label[:train_long]))
test_label = torch.from_numpy(np.array(all_label[train_long:]))

train_correlation = torch.from_numpy(np.array(correlation[:train_long]))
test_correlation = torch.from_numpy(np.array(correlation[train_long:]))

train_text_data = train_text_data.resize_(train_long, 1, 1024)
test_text_data = test_text_data.resize_(test_long, 1, 1024)
train_img_data = train_img_data.resize_(train_long, 1, 1024)
test_img_data = test_img_data.resize_(test_long, 1, 1024)
train_text2_data = train_text2_data.resize_(train_long, 1, 1024)
test_text2_data = test_text2_data.resize_(test_long, 1, 1024)
train_img2_data = train_img2_data.resize_(train_long, 1, 1024)
test_img2_data = test_img2_data.resize_(test_long, 1, 1024)

train_data = zip(train_text_data, train_text2_data, train_img_data, train_img2_data, train_label, train_correlation)
test_data = zip(test_text_data, test_text2_data, test_img_data, test_img2_data, test_label, test_correlation)
"""
def train_test_val_split(df,ratio_train,ratio_test,ratio_val):
    train, middle = train_test_split(df,test_size=1-ratio_train,random_state=0)
    ratio=ratio_val/(1-ratio_train)
    test,validation =train_test_split(middle,test_size=ratio,random_state=0)
    return train,test,validation


text_feature = torch.from_numpy(np.array(text_feature))
text_feature2 = torch.from_numpy(np.array(text_feature2))
img_feature = torch.from_numpy(np.array(img_feature))
img_feature2 = torch.from_numpy(np.array(img_feature2))

text_feature = text_feature.resize_(dataset_len, 1, 1024)
text_feature2 = text_feature2.resize_(dataset_len, 1, 1024)
img_feature = img_feature.resize_(dataset_len, 1, 1024)
img_feature2 = img_feature2.resize_(dataset_len, 1, 1024)


all_data = list(zip(text_feature, text_feature2, img_feature, img_feature2, all_label, correlation))


train_data, test_data, val_data= train_test_val_split(all_data, 0.8, 0.1, 0.1)
train_loader = data.DataLoader(list(train_data), batch_size=128, shuffle=False) #在分训练、测试、验证集的时候已经随机过了
test_loader = data.DataLoader(list(test_data), batch_size=128, shuffle=False)
val_loader = data.DataLoader(list(val_data), batch_size=128, shuffle=False)


print('数据加载完毕')
from collections import Counter

loss_func = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0,1.0,1.0])).float())#差15%
#loss_func = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.5,1.0,1.5])).float())#差的更多20%
#loss_func = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.5,1.0,0.5])).float())#差的超级多25%
#loss_func = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.8,1.0,1.2])).float())#差的超级超级多30%

opt = torch.optim.Adam(model.parameters(), weight_decay=0.0001, lr=learning_rate)
loss_count = []
acc_count = []
test_acc_count = []
best_val_acc = 0
best_val_acc_epoch = 0



#用MHFF
for epoch in range(300):
    print('_________________________________________________________________')
    print('epoch:', epoch)
    train_ans = []
    train_y_ture = []
    train_loss = 0

    for i, (t1, t2, i1, i2, y, c) in enumerate(train_loader):
        batch_t1 = Variable(t1)
        batch_t2 = Variable(t2)
        batch_i1 = Variable(i1)
        batch_i2 = Variable(i2)
        batch_y = Variable(y)
        # 获取最后输出
        batch_t1 = torch.tensor(batch_t1, dtype=torch.float32)
        batch_t2 = torch.tensor(batch_t2, dtype=torch.float32)
        batch_i1 = torch.tensor(batch_i1, dtype=torch.float32)
        batch_i2 = torch.tensor(batch_i2, dtype=torch.float32)

        #out = model(batch_t1, batch_t2, batch_i1, batch_i2, c) # torch.Size([128,3])
        out_layer1, out_layer2, out_layer3, total_out = model(batch_t1, batch_i1, c)  # torch.Size([128,3])
        out = total_out
        #out = (out_layer1 + out_layer2 + out_layer3)/3
        train_ans.extend(torch.max(out, 1)[1].numpy())

        # 获取损失
        batch_y = torch.tensor(batch_y, dtype=torch.long)
        train_y_ture.extend(batch_y.numpy())

        loss1 = loss_func(out_layer1, batch_y)
        loss2 = loss_func(out_layer2, batch_y)
        loss3 = loss_func(out_layer3, batch_y)
        loss4 = loss_func(total_out, batch_y)
        loss = loss1 + loss2 + loss3 + loss4
        #loss = loss1 + loss2 + loss3
        #loss = loss3
        train_loss = train_loss + loss

        # 使用优化器优化损失
        opt.zero_grad()  # 清空上一步残余更新参数值
        loss.backward() # 误差反向传播，计算参数更新值
        opt.step() # 将参数更新值施加到net的parmeters上
        if i == len(train_loader) - 1:
            loss_count.append(loss)
            print("loss: ", train_loss.item())
            accuracy = np.mean(train_ans == np.array(train_y_ture))
            print("accuracy: ", accuracy)
            print(Counter(train_ans))

            accuracy1 = []
            ans = []
            y_true = []
            test_loss = 0

            for t1_, t2_, i1_, i2_, y_, c_ in test_loader:
                test_t1 = Variable(t1_)
                test_t2 = Variable(t2_)
                test_i1 = Variable(i1_)
                test_i2 = Variable(i2_)
                test_y = Variable(y_)
                test_t1 = torch.tensor(test_t1, dtype=torch.float32)
                test_t2 = torch.tensor(test_t2, dtype=torch.float32)
                test_i1 = torch.tensor(test_i1, dtype=torch.float32)
                test_i2 = torch.tensor(test_i2, dtype=torch.float32)
                #out1 = model(test_t1, test_t2, test_i1, test_i2, c_)
                test_out_layer1, test_out_layer2, test_out_layer3, test_total_out = model(test_t1, test_i1, c_)
                #test_out = (test_out_layer1 + test_out_layer2 + test_out_layer3)/3
                test_out = test_total_out
                loss = loss_func(test_out, test_y)
                test_loss = test_loss + loss
                # print(torch.max(out, 1)[1].numpy())
                ans.extend(torch.max(test_out, 1)[1].numpy())
                test_y = torch.tensor(test_y, dtype=torch.long)
                y_true.extend(test_y.numpy())

            print(Counter(ans))
            print('test loss:\t', test_loss.item())

            print('test accuracy:\t', np.mean(ans == np.array(y_true)))
            f1_ = f1_score(ans, y_true, average='weighted')
            print('test weighted F1:\t', f1_)
            f1__ = f1_score(ans, y_true, average='micro')
            print('test micro F1:\t', f1__)
            f1___ = f1_score(ans, y_true, average='macro')
            print('test macro F1:\t', f1___)

            test_acc_count.append(accuracy1)
            if np.mean(ans == np.array(y_true)) > best_val_acc:
                np.save("text_img_test_softmax.npy", np.array(ans))
                torch.save(model.state_dict(), save_path + '.pkl')
                best_val_acc = np.mean(ans == np.array(y_true))
                best_val_F1 = f1_
                best_val_macro_f1 = f1___
                best_val_acc_epoch = epoch
                print("save!")
            print("best_val_acc_epoch : ", best_val_acc_epoch,'  best_val_acc:', best_val_acc, '  best_val_weighted_f1:'
                  , best_val_F1, '  best_val_macro_f1:', best_val_macro_f1)


"""

#只用gru对文本进行训练
for epoch in range(300):
    print('_________________________________________________________________')
    print('epoch:', epoch)
    for i, (t1, t2, i1, i2, y, c) in enumerate(train_loader):
        batch_t1 = Variable(t1)
        batch_i1 = Variable(i1)
        batch_y = Variable(y)
        # 获取最后输出
        batch_t1 = torch.tensor(batch_t1, dtype=torch.float32)
        batch_i1 = torch.tensor(batch_i1, dtype=torch.float32)

        out = model(batch_t1)

        # 获取损失
        batch_y = torch.tensor(batch_y, dtype=torch.long)
        loss = loss_func(out, batch_y)
        # 使用优化器优化损失
        opt.zero_grad()  # 清空上一步残余更新参数值
        loss.backward() # 误差反向传播，计算参数更新值
        opt.step() # 将参数更新值施加到net的parmeters上
        if i == 0:
            loss_count.append(loss)
            print("loss: ", loss.item())
            accuracy = np.mean(torch.max(out, 1)[1].numpy() == batch_y.numpy())
            print("accuracy: ", accuracy)
            acc_count.append(accuracy)
            print(Counter(torch.max(out, 1)[1].numpy()))

            accuracy1 = []
            ans = []
            y_true = []
            for t1_, t2_, i1_, i2_, y_, c_ in test_loader:
                test_t1 = Variable(t1_)
                test_i1 = Variable(i1_)
                test_y = Variable(y_)
                test_t1 = torch.tensor(test_t1, dtype=torch.float32)
                test_i1 = torch.tensor(test_i1, dtype=torch.float32)
                #out1 = model(test_t1, test_t2, test_i1, test_i2, c_)
                test_out = model(test_t1)
                ans.extend(torch.max(test_out, 1)[1].numpy())
                test_y = torch.tensor(test_y, dtype=torch.long)
                y_true.extend(test_y.numpy())


            print('test accuracy:\t', np.mean(ans == np.array(y_true)))
            f1_ = f1_score(ans, y_true, average='weighted')
            print('test weighted F1:\t', f1_)
            f1__ = f1_score(ans, y_true, average='micro')
            print('test micro F1:\t', f1__)
            f1___ = f1_score(ans, y_true, average='macro')
            print('test macro F1:\t', f1___)

            test_acc_count.append(accuracy1)
            if np.mean(ans == np.array(y_true)) > best_val_acc:
                np.save("text_img_test_softmax.npy", np.array(ans))
                torch.save(model.state_dict(), save_path + '.pkl')
                best_val_acc = np.mean(ans == np.array(y_true))
                best_val_F1 = f1_
                best_val_macro_f1 = f1___
                best_val_acc_epoch = epoch
                print("save!")
            print("best_val_acc_epoch : ", best_val_acc_epoch, '  best_val_acc:', best_val_acc,
                  '  best_val_weighted_f1:'
                  , best_val_F1, '  best_val_macro_f1:', best_val_macro_f1)

"""
"""
#只用CNN对图片进行训练
for epoch in range(300):
    print('_________________________________________________________________')
    print('epoch:', epoch)
    for i, (t1, t2, i1, i2, y, c) in enumerate(train_loader):
        batch_t1 = Variable(t1)
        batch_i1 = Variable(i1)
        batch_y = Variable(y)
        # 获取最后输出
        batch_t1 = torch.tensor(batch_t1, dtype=torch.float32)
        batch_i1 = torch.tensor(batch_i1, dtype=torch.float32)

        out = model(batch_i1)

        # 获取损失
        batch_y = torch.tensor(batch_y, dtype=torch.long)
        loss = loss_func(out, batch_y)
        # 使用优化器优化损失
        opt.zero_grad()  # 清空上一步残余更新参数值
        loss.backward() # 误差反向传播，计算参数更新值
        opt.step() # 将参数更新值施加到net的parmeters上
        if i == 0:
            loss_count.append(loss)
            print("loss: ", loss.item())
            accuracy = np.mean(torch.max(out, 1)[1].numpy() == batch_y.numpy())
            print("accuracy: ", accuracy)
            acc_count.append(accuracy)
            print(Counter(torch.max(out, 1)[1].numpy()))

            accuracy1 = []
            ans = []
            y_true = []
            for t1_, t2_, i1_, i2_, y_, c_ in test_loader:
                test_t1 = Variable(t1_)
                test_i1 = Variable(i1_)
                test_y = Variable(y_)
                test_t1 = torch.tensor(test_t1, dtype=torch.float32)
                test_i1 = torch.tensor(test_i1, dtype=torch.float32)
                #out1 = model(test_t1, test_t2, test_i1, test_i2, c_)
                test_out = model(test_i1)
                ans.extend(torch.max(test_out, 1)[1].numpy())
                test_y = torch.tensor(test_y, dtype=torch.long)
                y_true.extend(test_y.numpy())



            print('test accuracy:\t', np.mean(ans == np.array(y_true)))
            f1_ = f1_score(ans, y_true, average='weighted')
            print('test weighted F1:\t', f1_)
            f1__ = f1_score(ans, y_true, average='micro')
            print('test micro F1:\t', f1__)
            f1___ = f1_score(ans, y_true, average='macro')
            print('test macro F1:\t', f1___)

            test_acc_count.append(accuracy1)
            if np.mean(ans == np.array(y_true)) > best_val_acc:
                np.save("text_img_test_softmax.npy", np.array(ans))
                torch.save(model.state_dict(), save_path + '.pkl')
                best_val_acc = np.mean(ans == np.array(y_true))
                best_val_F1 = f1_
                best_val_macro_f1 = f1___
                best_val_acc_epoch = epoch
                print("save!")
            print("best_val_acc_epoch : ", best_val_acc_epoch, '  best_val_acc:', best_val_acc,
                  '  best_val_weighted_f1:'
                  , best_val_F1, '  best_val_macro_f1:', best_val_macro_f1)
"""
"""
#不用MHFF

for epoch in range(300):
    print('_________________________________________________________________')
    print('epoch:', epoch)
    train_ans = []
    train_y_ture = []
    train_loss = 0
    for i, (t1, t2, i1, i2, y, c) in enumerate(train_loader):
        batch_t1 = Variable(t1)
        batch_t2 = Variable(t2)
        #batch_t2 = Variable(t1)
        batch_i1 = Variable(i1)
        batch_i2 = Variable(i2)
        #batch_i2 = Variable(i1)
        batch_y = Variable(y)
        # 获取最后输出
        batch_t1 = torch.tensor(batch_t1, dtype=torch.float32)
        batch_t2 = torch.tensor(batch_t2, dtype=torch.float32)
        batch_i1 = torch.tensor(batch_i1, dtype=torch.float32)
        batch_i2 = torch.tensor(batch_i2, dtype=torch.float32)

        out = model(batch_t1, batch_t2, batch_i1, batch_i2, c) # torch.Size([128,3])

        train_ans.extend(torch.max(out, 1)[1].numpy())


        # 获取损失
        batch_y = torch.tensor(batch_y, dtype=torch.long)
        train_y_ture.extend(batch_y.numpy())
        loss5 = loss_func(out, batch_y)

        #loss = loss1 + loss2 + loss3 + loss4
        #loss = loss1 + loss2 + loss3
        loss = loss5
        train_loss = train_loss + loss

        # 使用优化器优化损失
        opt.zero_grad()  # 清空上一步残余更新参数值
        loss.backward() # 误差反向传播，计算参数更新值
        opt.step() # 将参数更新值施加到net的parmeters上
        if i == len(train_loader) - 1:
            print("loss: ", train_loss.item())
            accuracy = np.mean(train_ans == np.array(train_y_ture))
            print("accuracy: ", accuracy)
            acc_count.append(accuracy)
            print(Counter(train_ans))

            accuracy1 = []
            ans = []
            y_true = []
            test_loss = 0
            for t1_, t2_, i1_, i2_, y_, c_ in test_loader:
                test_t1 = Variable(t1_)
                test_t2 = Variable(t2_)
                #test_t2 = Variable(t1_)
                test_i1 = Variable(i1_)
                test_i2 = Variable(i2_)
                #test_i2 = Variable(i1_)
                test_y = Variable(y_)
                test_t1 = torch.tensor(test_t1, dtype=torch.float32)
                test_t2 = torch.tensor(test_t2, dtype=torch.float32)
                test_i1 = torch.tensor(test_i1, dtype=torch.float32)
                test_i2 = torch.tensor(test_i2, dtype=torch.float32)
                out1 = model(test_t1, test_t2, test_i1, test_i2, c_)

                loss = loss_func(out1, test_y)
                test_loss = test_loss + loss
                ans.extend(torch.max(out1, 1)[1].numpy())
                test_y = torch.tensor(test_y, dtype=torch.long)
                y_true.extend(test_y.numpy())

            print('test loss:\t', test_loss.item())

            print('test accuracy:\t', np.mean(ans == np.array(y_true)))
            f1_ = f1_score(ans, y_true, average='weighted')
            print('test weighted F1:\t', f1_)
            f1__ = f1_score(ans, y_true, average='micro')
            print('test micro F1:\t', f1__)
            f1___ = f1_score(ans, y_true, average='macro')
            print('test macro F1:\t', f1___)

            test_acc_count.append(accuracy1)
            if np.mean(ans == np.array(y_true)) > best_val_acc:
                np.save("text_img_test_softmax.npy", np.array(ans))
                torch.save(model.state_dict(), save_path + '.pkl')
                best_val_acc = np.mean(ans == np.array(y_true))
                best_val_F1 = f1_
                best_val_macro_f1 = f1___
                best_val_acc_epoch = epoch
                print("save!")
            print("best_val_acc_epoch : ", best_val_acc_epoch,'  best_val_acc:', best_val_acc, '  best_val_weighted_f1:'
                  , best_val_F1, '  best_val_macro_f1:', best_val_macro_f1)

"""