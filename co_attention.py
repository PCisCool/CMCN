from fc import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F

DROPOUT_R = 0.7


MFB_K = 3
MFB_O = 100

LSTM_OUT_SIZE = 512
FRCN_FEAT_SIZE = 512
HIDDEN_SIZE = 128
ouput_mid = 50
#ouput_mid2 = 20
Q_GLIMPSES = 2
I_GLIMPSES = 2

fus_ = 1.5


class MFB(nn.Module):
    def __init__(self, img_feat_size, ques_feat_size):
        super(MFB, self).__init__()
        self.proj_i = nn.Linear(img_feat_size, MFB_K * MFB_O)    # __C.MFB_K * __C.MFB_O
        self.proj_q = nn.Linear(ques_feat_size, MFB_K * MFB_O)   #__C.MFB_K * __C.MFB_O
        self.dropout = nn.Dropout(DROPOUT_R)
        self.pool = nn.AvgPool1d(MFB_K, stride=MFB_K)

    def forward(self, img_feat, ques_feat):
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = torch.tensor(ques_feat, dtype=torch.float32)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)

        exp_out = img_feat * ques_feat                  # (N, C, K*O)
        exp_out = self.dropout(exp_out)# (N, C, K*O)
        z = self.pool(exp_out) * MFB_K         # (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))         # (N, C*O)
        z = z.view(batch_size, -1, MFB_O)      # (N, C, O)
        return z, exp_out

class fus(nn.Module):
    def __init__(self, img_feat_size, ques_feat_size):
        super(fus, self).__init__()
        self.proj_i = nn.Linear(img_feat_size, MFB_K * MFB_O)    # __C.MFB_K * __C.MFB_O
        self.proj_q = nn.Linear(ques_feat_size, MFB_K * MFB_O)   #__C.MFB_K * __C.MFB_O
        self.dropout = nn.Dropout(DROPOUT_R)
        self.pool = nn.AvgPool1d(MFB_K, stride=MFB_K)

    def forward(self, img_feat, ques_feat):
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = torch.tensor(ques_feat, dtype=torch.float32)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)

        exp_out = img_feat + fus_ * ques_feat                  # (N, C, K*O)
        #exp_out = img_feat * ques_feat
        #exp_out = img_feat
        exp_out = self.dropout(exp_out)# (N, C, K*O)
        z = self.pool(exp_out) * MFB_K         # (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))         # (N, C*O)
        z = z.view(batch_size, -1, MFB_O)      # (N, C, O)
        return z, exp_out

class QAtt(nn.Module):
    def __init__(self):
        super(QAtt, self).__init__()
        self.mlp = MLP(
            in_size=LSTM_OUT_SIZE,
            mid_size=HIDDEN_SIZE,
            out_size=Q_GLIMPSES,
            dropout_r=DROPOUT_R,
            use_relu=True
        )

    def forward(self, ques_feat):
        '''
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            qatt_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
        '''
        qatt_maps = self.mlp(ques_feat)                 # (N, T, Q_GLIMPSES)
        qatt_maps = F.softmax(qatt_maps, dim=1)         # (N, T, Q_GLIMPSES)

        qatt_feat_list = []
        for i in range(Q_GLIMPSES):
            mask = qatt_maps[:, :, i:i + 1]             # (N, T, 1)
            mask = mask * ques_feat                     # (N, T, LSTM_OUT_SIZE)
            mask = torch.sum(mask, dim=1)               # (N, LSTM_OUT_SIZE)
            qatt_feat_list.append(mask)
        qatt_feat = torch.cat(qatt_feat_list, dim=1)    # (N, LSTM_OUT_SIZE*Q_GLIMPSES)

        return qatt_feat


class IAtt(nn.Module):
    def __init__(self, img_feat_size, ques_att_feat_size):
        super(IAtt, self).__init__()
        self.dropout = nn.Dropout(DROPOUT_R)
        self.mfb = MFB(img_feat_size, ques_att_feat_size)
        self.img_feat_fus = fus(img_feat_size, ques_att_feat_size)
        self.mlp = MLP(
            in_size=MFB_O,
            mid_size=HIDDEN_SIZE,
            out_size=I_GLIMPSES,
            dropout_r=DROPOUT_R,
            use_relu=True
        )

    def forward(self, img_feat, ques_att_feat, correlation):
        '''
            img_feats.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_att_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
            iatt_feat.size() -> (N, MFB_O * I_GLIMPSES)
        '''
        #ques_att_feat = torch.transpose(ques_att_feat, 1, 0)
        #ques_att_feat = correlation * ques_att_feat
        #ques_att_feat = torch.transpose(ques_att_feat, 1, 0)
        ques_att_feat = ques_att_feat.unsqueeze(1)      # (N, 1, LSTM_OUT_SIZE * Q_GLIMPSES)
        img_feat = self.dropout(img_feat)
        #z, _ = self.mfb(img_feat, ques_att_feat)        # (N, C, O)
        z, _ = self.img_feat_fus(img_feat, ques_att_feat)  # (N, C, O)

        iatt_maps = self.mlp(z)                         # (N, C, I_GLIMPSES)
        iatt_maps = F.softmax(iatt_maps, dim=1)         # (N, C, I_GLIMPSES)

        iatt_feat_list = []
        for i in range(I_GLIMPSES):
            mask = iatt_maps[:, :, i:i + 1]             # (N, C, 1)
            mask = mask * img_feat                      # (N, C, FRCN_FEAT_SIZE)
            mask = torch.sum(mask, dim=1)               # (N, FRCN_FEAT_SIZE)
            iatt_feat_list.append(mask)
        iatt_feat = torch.cat(iatt_feat_list, dim=1)    # (N, FRCN_FEAT_SIZE*I_GLIMPSES)

        return iatt_feat


class IAtt_NOc(nn.Module):
    def __init__(self, img_feat_size, ques_att_feat_size):
        super(IAtt_NOc, self).__init__()
        self.dropout = nn.Dropout(DROPOUT_R)
        self.mfb = MFB(img_feat_size, ques_att_feat_size)
        self.img_feat_fus = fus(img_feat_size, ques_att_feat_size)
        self.mlp = MLP(
            in_size=MFB_O,
            mid_size=HIDDEN_SIZE,
            out_size=I_GLIMPSES,
            dropout_r=DROPOUT_R,
            use_relu=True
        )

    def forward(self, img_feat, ques_att_feat):
        '''
            img_feats.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_att_feat.size() -> (N, LSTM_OUT_SIZE * Q_GLIMPSES)
            iatt_feat.size() -> (N, MFB_O * I_GLIMPSES)
        '''
        ques_att_feat = ques_att_feat.unsqueeze(1)    # (N, 1, LSTM_OUT_SIZE * Q_GLIMPSES)
        img_feat = self.dropout(img_feat)
        #z, _ = self.mfb(img_feat, ques_att_feat)        # (N, C, O)
        z, _ = self.img_feat_fus(img_feat, ques_att_feat)  # (N, C, O)

        iatt_maps = self.mlp(z)                         # (N, C, I_GLIMPSES)
        iatt_maps = F.softmax(iatt_maps, dim=1)         # (N, C, I_GLIMPSES)

        iatt_feat_list = []
        for i in range(I_GLIMPSES):
            mask = iatt_maps[:, :, i:i + 1]             # (N, C, 1)
            mask = mask * img_feat                      # (N, C, FRCN_FEAT_SIZE)
            mask = torch.sum(mask, dim=1)               # (N, FRCN_FEAT_SIZE)
            iatt_feat_list.append(mask)
        iatt_feat = torch.cat(iatt_feat_list, dim=1)    # (N, FRCN_FEAT_SIZE*I_GLIMPSES)

        return iatt_feat


class CoAtt(nn.Module):
    def __init__(self):
        super(CoAtt, self).__init__()

        img_feat_size = LSTM_OUT_SIZE
        img_att_feat_size = img_feat_size * I_GLIMPSES
        ques_att_feat_size = LSTM_OUT_SIZE * Q_GLIMPSES

        self.q_att = QAtt()
        self.i_att = IAtt(img_feat_size, ques_att_feat_size)

        self.mfb = MFB(img_att_feat_size, ques_att_feat_size)
        self.output = nn.Linear(MFB_O, 3)


    def forward(self, img_feat, ques_feat, correlation):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        ques_feat = ques_feat.view(ques_feat.size(0), int(1024 / LSTM_OUT_SIZE),
                                   LSTM_OUT_SIZE)  # 更改维度
        img_feat = img_feat.view(img_feat.size(0), int(1024 / FRCN_FEAT_SIZE),
                                 FRCN_FEAT_SIZE)
        ques_feat = self.q_att(ques_feat)               # (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        fuse_feat = self.i_att(img_feat, ques_feat ,correlation)     # (N, FRCN_FEAT_SIZE*I_GLIMPSES)


        z, _ = self.mfb(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))             # z:(N, 1, O)  _:(N, C, K*O)
        z = z.squeeze(1)                                                            # (N, O)
        z = self.output(z)

        return z


class double_CoAtt(nn.Module):
    def __init__(self):
        super(double_CoAtt, self).__init__()

        img_feat_size = LSTM_OUT_SIZE
        img_att_feat_size = img_feat_size * I_GLIMPSES
        ques_att_feat_size = LSTM_OUT_SIZE * Q_GLIMPSES

        self.q_att = QAtt()
        self.i_att = IAtt(img_feat_size, ques_att_feat_size)

        self.mfb = MFB(img_att_feat_size, ques_att_feat_size)
        self.output = nn.Linear(MFB_O, 3)


    def forward(self, text_feat, text_feat2, img_feat, img_feat2, correlation):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        text_feature_cat = torch.cat([text_feat,text_feat2],dim=2)
        img_feature_cat = torch.cat([img_feat,img_feat2],dim=2)

        text_feat = text_feature_cat.view(text_feature_cat.size(0), int(2048 / LSTM_OUT_SIZE),
                                   LSTM_OUT_SIZE)  # 更改维度
        img_feat = img_feature_cat.view(img_feature_cat.size(0), int(2048 / FRCN_FEAT_SIZE),
                                 FRCN_FEAT_SIZE)

        text_feat = self.q_att(text_feat)               # (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        fuse_feat = self.i_att(img_feat, text_feat ,correlation)     # (N, FRCN_FEAT_SIZE*I_GLIMPSES)


        z, _ = self.mfb(fuse_feat.unsqueeze(1), text_feat.unsqueeze(1))             # z:(N, 1, O)  _:(N, C, K*O)
        z = z.squeeze(1)                                                            # (N, O)
        z = self.output(z)

        return z


class treble_CoAtt(nn.Module):
    def __init__(self):
        super(treble_CoAtt, self).__init__()

        img_feat_size = LSTM_OUT_SIZE
        img_att_feat_size = img_feat_size * I_GLIMPSES
        ques_att_feat_size = LSTM_OUT_SIZE * Q_GLIMPSES

        self.q_att = QAtt()
        self.i_att = IAtt(img_feat_size, ques_att_feat_size)

        self.mfb = MFB(img_att_feat_size, ques_att_feat_size)
        self.output = nn.Linear(MFB_O, 3)


    def forward(self, text_feat, text_feat2, text_feat3, img_feat, img_feat2, img_feat3, correlation):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        text_feature_cat = torch.cat([text_feat,text_feat2,text_feat3],dim=2)
        img_feature_cat = torch.cat([img_feat,img_feat2,img_feat3],dim=2)

        text_feat = text_feature_cat.view(text_feature_cat.size(0), int(3072 / LSTM_OUT_SIZE),
                                   LSTM_OUT_SIZE)  # 更改维度
        img_feat = img_feature_cat.view(img_feature_cat.size(0), int(3072 / FRCN_FEAT_SIZE),
                                 FRCN_FEAT_SIZE)

        text_feat = self.q_att(text_feat)               # (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        fuse_feat = self.i_att(img_feat, text_feat ,correlation)     # (N, FRCN_FEAT_SIZE*I_GLIMPSES)


        z, _ = self.mfb(fuse_feat.unsqueeze(1), text_feat.unsqueeze(1))             # z:(N, 1, O)  _:(N, C, K*O)
        z = z.squeeze(1)                                                            # (N, O)
        z = self.output(z)

        return z

from CNN import CNNnet, multi_CNNnet, CNNnet_ZOL, CNNnet_ZOL_10_onlyImg2
from RNN import RNNnet, multi_RNNnet, RNNnet_ZOL, RNNnet_ZOL_10_only_text2

class CoAtt_and_Independent(nn.Module):
    def __init__(self):
        super(CoAtt_and_Independent, self).__init__()

        img_feat_size = LSTM_OUT_SIZE
        img_att_feat_size = img_feat_size * I_GLIMPSES
        ques_att_feat_size = LSTM_OUT_SIZE * Q_GLIMPSES
        self.zip = nn.Linear(1024, MFB_O)

        self.q_att = QAtt()
        self.i_att = IAtt(img_feat_size, ques_att_feat_size)

        self.mfb = MFB(img_att_feat_size, ques_att_feat_size)
        self.output1 = torch.nn.Sequential(nn.Linear(3 * MFB_O, 3),
                                torch.nn.ReLU())
        #self.output1 = torch.nn.Sequential(nn.Linear(MFB_O, ouput_mid),
        #                                   torch.nn.ReLU())
        #self.output2 = torch.nn.Sequential(nn.Linear(ouput_mid, 3),
        #                        torch.nn.ReLU())
        self.total_out = nn.Linear(9, 3)




    def forward(self, ques_feat, text_, img_feat, img_,  correlation):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        model1 =RNNnet()
        model2 =CNNnet()
        text = model1(text_)
        img_feature = img_.resize_(img_.size(0), 1, 32, 32)
        img = model2(img_feature)
        text2 = self.zip(ques_feat).squeeze(1)
        img2 = self.zip(img_feat).squeeze(1)
        ques_feat = ques_feat.view(ques_feat.size(0), int(1024 / LSTM_OUT_SIZE),
                                   LSTM_OUT_SIZE)  # 更改维度
        img_feat = img_feat.view(img_feat.size(0), int(1024 / FRCN_FEAT_SIZE),
                                 FRCN_FEAT_SIZE)
        ques_feat = self.q_att(ques_feat)               # (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        fuse_feat = self.i_att(img_feat, ques_feat ,correlation)     # (N, FRCN_FEAT_SIZE*I_GLIMPSES)


        z, _ = self.mfb(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))             # z:(N, 1, O)  _:(N, C, K*O)
        z = z.squeeze(1)                                                            # (N, O)
        z = torch.cat([z,text2,img2],dim=1)
        z = self.output1(z)
        #z = self.output2(z)


        total = torch.cat([text,img],dim=1)
        total = torch.cat([total,z],dim=1)
        total = self.total_out(total)

        return total


class multi_CoAtt_and_Independent(nn.Module):
    def __init__(self):
        super(multi_CoAtt_and_Independent, self).__init__()

        img_feat_size = LSTM_OUT_SIZE
        img_att_feat_size = img_feat_size * I_GLIMPSES
        ques_att_feat_size = LSTM_OUT_SIZE * Q_GLIMPSES

        self.zip = nn.Linear(1024, MFB_O)

        self.q_att = QAtt()
        self.i_att = IAtt(img_feat_size, ques_att_feat_size)

        self.mfb = MFB(img_att_feat_size, ques_att_feat_size)
        self.output1 = torch.nn.Sequential(nn.Linear(3*MFB_O, ouput_mid),
                                torch.nn.ReLU())
        self.output2 = torch.nn.Sequential(nn.Linear(ouput_mid, 3),
                                torch.nn.ReLU())
        self.total_out = nn.Linear(9, 3)




    def forward(self, ques_feat, text_, img_feat, img_,  correlation):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        model1 =multi_RNNnet()
        model2 =multi_CNNnet()
        #model3 =multi_RNNnet()
        #model4 =multi_CNNnet()
        text = model1(text_)
        img_feature = img_.resize_(img_.size(0), 1, 32, 32)
        img = model2(img_feature)

        text2 = self.zip(ques_feat).squeeze(1)
        img2 = self.zip(img_feat).squeeze(1)

        ques_feat = ques_feat.view(ques_feat.size(0), int(1024 / LSTM_OUT_SIZE),
                                   LSTM_OUT_SIZE)  # 更改维度
        img_feat = img_feat.view(img_feat.size(0), int(1024 / FRCN_FEAT_SIZE),
                                 FRCN_FEAT_SIZE)
        ques_feat = self.q_att(ques_feat)               # (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        fuse_feat = self.i_att(img_feat, ques_feat ,correlation)     # (N, FRCN_FEAT_SIZE*I_GLIMPSES)


        z, _ = self.mfb(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))             # z:(N, 1, O)  _:(N, C, K*O)
        z = z.squeeze(1)                                                            # (N, O)
        z = torch.cat([z,text2,img2],dim=1)
        z = self.output1(z)
        z = self.output2(z)
        #z = self.output3(z)





        total = torch.cat([text,img,z],dim=1)
        total = self.total_out(total)

        return total




class MHC(nn.Module):           #3分类
    def __init__(self, ques_feat_size, ques_att_feat_size, img_feat_size, img_att_feat_size ):
        super(MHC, self).__init__()
        self.proj_i = nn.Linear(img_feat_size, MFB_K * MFB_O)    # __C.MFB_K * __C.MFB_O
        self.proj_q = nn.Linear(ques_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.proj_ia = nn.Linear(img_att_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.proj_qa = nn.Linear(ques_att_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.dropout = nn.Dropout(DROPOUT_R)
        self.pool = nn.AvgPool1d(MFB_K, stride=MFB_K)

        self.linear1 = torch.nn.Sequential(torch.nn.Linear(4 * MFB_O, 3),
                                        torch.nn.Softmax())
        self.linear2 = torch.nn.Sequential(torch.nn.Linear(MFB_O, 3),
                                        torch.nn.Softmax())
        self.linear3 = torch.nn.Sequential(torch.nn.Linear(3 * 3, 3),     #3层，每层的输出是3
                                        torch.nn.Softmax())

    def forward(self, ques_feat, ques_att_feat, img_feat, img_att_feat):
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = torch.tensor(ques_feat, dtype=torch.float32)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)
        ques_att_feat = self.proj_qa(ques_att_feat)
        img_att_feat = self.proj_ia(img_att_feat)

        #第一层
        feat_layer1 = torch.cat([ques_feat, img_feat, ques_att_feat, img_att_feat], dim=2)
        feat_layer1 = self.dropout(feat_layer1)
        feat_layer1 = self.pool(feat_layer1) * MFB_K         # (N, C, O)
        feat_layer1 = torch.sqrt(F.relu(feat_layer1)) - torch.sqrt(F.relu(-feat_layer1))
        feat_layer1 = F.normalize(feat_layer1.view(batch_size, -1))         # (N, C*O)
        feat_layer1 = feat_layer1.view(batch_size, 1, 4 * MFB_O)
        out_layer1 = self.linear1(feat_layer1)

        #第二层
        fus_feat1 = ques_feat * img_feat
        fus_feat2 = ques_att_feat * img_att_feat
        fus_feat3 = ques_feat * ques_att_feat
        fus_feat4 = img_feat * img_att_feat

        feat_layer2 = torch.cat([fus_feat1, fus_feat2, fus_feat3, fus_feat4], dim=2)
        feat_layer2 = self.dropout(feat_layer2)
        feat_layer2 = self.pool(feat_layer2) * MFB_K         # (N, C, O)
        feat_layer2 = torch.sqrt(F.relu(feat_layer2)) - torch.sqrt(F.relu(-feat_layer2))
        feat_layer2 = F.normalize(feat_layer2.view(batch_size, -1))         # (N, C*O)
        feat_layer2 = feat_layer2.view(batch_size, 1, 4 * MFB_O)
        out_layer2 = self.linear1(feat_layer2)

        # 第三层
        all_fus_feat = fus_feat1 * fus_feat2 * fus_feat3 * fus_feat4

        feat_layer3 = self.dropout(all_fus_feat)
        feat_layer3 = self.pool(feat_layer3) * MFB_K  # (N, C, O)
        feat_layer3 = torch.sqrt(F.relu(feat_layer3)) - torch.sqrt(F.relu(-feat_layer3))
        feat_layer3 = F.normalize(feat_layer3.view(batch_size, -1))  # (N, C*O)
        feat_layer3 = feat_layer3.view(batch_size, 1, MFB_O)
        out_layer3 = self.linear2(feat_layer3)

        total_out = torch.cat([out_layer1, out_layer2, out_layer3], dim=2)
        total_out = self.linear3(total_out)

        return out_layer1, out_layer2, out_layer3, total_out


class MHC_no_cross_modal_fusion_layer(nn.Module):           #3分类
    def __init__(self, ques_feat_size, ques_att_feat_size, img_feat_size, img_att_feat_size ):
        super(MHC_no_cross_modal_fusion_layer, self).__init__()
        self.proj_i = nn.Linear(img_feat_size, MFB_K * MFB_O)    # __C.MFB_K * __C.MFB_O
        self.proj_q = nn.Linear(ques_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.proj_ia = nn.Linear(img_att_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.proj_qa = nn.Linear(ques_att_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.dropout = nn.Dropout(DROPOUT_R)
        self.pool = nn.AvgPool1d(MFB_K, stride=MFB_K)

        self.linear1 = torch.nn.Sequential(torch.nn.Linear(4 * MFB_O, 3),
                                        torch.nn.Softmax())
        self.linear2 = torch.nn.Sequential(torch.nn.Linear(MFB_O, 3),
                                        torch.nn.Softmax())
        self.linear3 = torch.nn.Sequential(torch.nn.Linear(2 * 3, 3),     #3层，每层的输出是3
                                        torch.nn.Softmax())

    def forward(self, ques_feat, ques_att_feat, img_feat, img_att_feat):
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = torch.tensor(ques_feat, dtype=torch.float32)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)
        ques_att_feat = self.proj_qa(ques_att_feat)
        img_att_feat = self.proj_ia(img_att_feat)

        #第一层
        feat_layer1 = torch.cat([ques_feat, img_feat, ques_att_feat, img_att_feat], dim=2)
        feat_layer1 = self.dropout(feat_layer1)
        feat_layer1 = self.pool(feat_layer1) * MFB_K         # (N, C, O)
        feat_layer1 = torch.sqrt(F.relu(feat_layer1)) - torch.sqrt(F.relu(-feat_layer1))
        feat_layer1 = F.normalize(feat_layer1.view(batch_size, -1))         # (N, C*O)
        feat_layer1 = feat_layer1.view(batch_size, 1, 4 * MFB_O)
        out_layer1 = self.linear1(feat_layer1)

        #第二层
        all_fus_feat = ques_feat * img_feat * ques_att_feat * img_att_feat

        feat_layer3 = self.dropout(all_fus_feat)
        feat_layer3 = self.pool(feat_layer3) * MFB_K  # (N, C, O)
        feat_layer3 = torch.sqrt(F.relu(feat_layer3)) - torch.sqrt(F.relu(-feat_layer3))
        feat_layer3 = F.normalize(feat_layer3.view(batch_size, -1))  # (N, C*O)
        feat_layer3 = feat_layer3.view(batch_size, 1, MFB_O)
        out_layer3 = self.linear2(feat_layer3)

        total_out = torch.cat([out_layer1, out_layer3], dim=2)
        total_out = self.linear3(total_out)

        return out_layer1, total_out, out_layer3, total_out


class MHC_no_global_fusion_layer(nn.Module):           #3分类
    def __init__(self, ques_feat_size, ques_att_feat_size, img_feat_size, img_att_feat_size ):
        super(MHC_no_global_fusion_layer, self).__init__()
        self.proj_i = nn.Linear(img_feat_size, MFB_K * MFB_O)    # __C.MFB_K * __C.MFB_O
        self.proj_q = nn.Linear(ques_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.proj_ia = nn.Linear(img_att_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.proj_qa = nn.Linear(ques_att_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.dropout = nn.Dropout(DROPOUT_R)
        self.pool = nn.AvgPool1d(MFB_K, stride=MFB_K)

        self.linear1 = torch.nn.Sequential(torch.nn.Linear(4 * MFB_O, 3),
                                        torch.nn.Softmax())
        self.linear2 = torch.nn.Sequential(torch.nn.Linear(MFB_O, 3),
                                        torch.nn.Softmax())
        self.linear3 = torch.nn.Sequential(torch.nn.Linear(3 * 3, 3),     #3层，每层的输出是3
                                        torch.nn.Softmax())

    def forward(self, ques_feat, ques_att_feat, img_feat, img_att_feat):
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = torch.tensor(ques_feat, dtype=torch.float32)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)
        ques_att_feat = self.proj_qa(ques_att_feat)
        img_att_feat = self.proj_ia(img_att_feat)

        #第一层
        feat_layer1 = torch.cat([ques_feat, img_feat, ques_att_feat, img_att_feat], dim=2)
        feat_layer1 = self.dropout(feat_layer1)
        feat_layer1 = self.pool(feat_layer1) * MFB_K         # (N, C, O)
        feat_layer1 = torch.sqrt(F.relu(feat_layer1)) - torch.sqrt(F.relu(-feat_layer1))
        feat_layer1 = F.normalize(feat_layer1.view(batch_size, -1))         # (N, C*O)
        feat_layer1 = feat_layer1.view(batch_size, 1, 4 * MFB_O)
        out_layer1 = self.linear1(feat_layer1)

        #第二层
        fus_feat1 = ques_feat * img_feat
        fus_feat2 = ques_att_feat * img_att_feat
        fus_feat3 = ques_feat * ques_att_feat
        fus_feat4 = img_feat * img_att_feat

        feat_layer2 = torch.cat([fus_feat1, fus_feat2, fus_feat3, fus_feat4], dim=2)
        feat_layer2 = self.dropout(feat_layer2)
        feat_layer2 = self.pool(feat_layer2) * MFB_K         # (N, C, O)
        feat_layer2 = torch.sqrt(F.relu(feat_layer2)) - torch.sqrt(F.relu(-feat_layer2))
        feat_layer2 = F.normalize(feat_layer2.view(batch_size, -1))         # (N, C*O)
        feat_layer2 = feat_layer2.view(batch_size, 1, 4 * MFB_O)
        out_layer2 = self.linear1(feat_layer2)

        return out_layer1, out_layer2, out_layer1, out_layer2

class MHC2(nn.Module):              #5分类
    def __init__(self, ques_feat_size, ques_att_feat_size, img_feat_size, img_att_feat_size ):
        super(MHC2, self).__init__()
        self.proj_i = nn.Linear(img_feat_size, MFB_K * MFB_O)    # __C.MFB_K * __C.MFB_O
        self.proj_q = nn.Linear(ques_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.proj_ia = nn.Linear(img_att_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.proj_qa = nn.Linear(ques_att_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.dropout = nn.Dropout(DROPOUT_R)
        self.pool = nn.AvgPool1d(MFB_K, stride=MFB_K)

        self.linear1 = torch.nn.Sequential(torch.nn.Linear(4 * MFB_O, 5)
                                           #, torch.nn.Softmax()
                                           )
        self.linear2 = torch.nn.Sequential(torch.nn.Linear(MFB_O, 5)
                                           #, torch.nn.Softmax()
                                           )
        self.linear3 = torch.nn.Sequential(torch.nn.Linear(3 * 5, 5)
                                           #, torch.nn.Softmax()
                                           )

    def forward(self, ques_feat, ques_att_feat, img_feat, img_att_feat):
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = torch.tensor(ques_feat, dtype=torch.float32)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)
        ques_att_feat = self.proj_qa(ques_att_feat)
        img_att_feat = self.proj_ia(img_att_feat)

        #第一层
        feat_layer1 = torch.cat([ques_feat, img_feat, ques_att_feat, img_att_feat], dim=2)
        feat_layer1 = self.dropout(feat_layer1)
        feat_layer1 = self.pool(feat_layer1) * MFB_K         # (N, C, O)
        feat_layer1 = torch.sqrt(F.relu(feat_layer1)) - torch.sqrt(F.relu(-feat_layer1))
        feat_layer1 = F.normalize(feat_layer1.view(batch_size, -1))         # (N, C*O)
        feat_layer1 = feat_layer1.view(batch_size, 1, 4 * MFB_O)
        out_layer1 = self.linear1(feat_layer1)

        #第二层
        fus_feat1 = ques_feat * img_feat
        fus_feat2 = ques_att_feat * img_att_feat
        fus_feat3 = ques_feat * ques_att_feat
        fus_feat4 = img_feat * img_att_feat

        feat_layer2 = torch.cat([fus_feat1, fus_feat2, fus_feat3, fus_feat4], dim=2)
        feat_layer2 = self.dropout(feat_layer2)
        feat_layer2 = self.pool(feat_layer2) * MFB_K         # (N, C, O)
        feat_layer2 = torch.sqrt(F.relu(feat_layer2)) - torch.sqrt(F.relu(-feat_layer2))
        feat_layer2 = F.normalize(feat_layer2.view(batch_size, -1))         # (N, C*O)
        feat_layer2 = feat_layer2.view(batch_size, 1, 4 * MFB_O)
        out_layer2 = self.linear1(feat_layer2)

        # 第三层
        all_fus_feat = fus_feat1 * fus_feat2 * fus_feat3 * fus_feat4

        feat_layer3 = self.dropout(all_fus_feat)
        feat_layer3 = self.pool(feat_layer3) * MFB_K  # (N, C, O)
        feat_layer3 = torch.sqrt(F.relu(feat_layer3)) - torch.sqrt(F.relu(-feat_layer3))
        feat_layer3 = F.normalize(feat_layer3.view(batch_size, -1))  # (N, C*O)
        feat_layer3 = feat_layer3.view(batch_size, 1, MFB_O)
        out_layer3 = self.linear2(feat_layer3)

        total_out = torch.cat([out_layer1, out_layer2, out_layer3], dim=2)
        total_out = self.linear3(total_out)

        return out_layer1, out_layer2, out_layer3, total_out



class MHC3(nn.Module):                  #10分类
    def __init__(self, ques_feat_size, ques_att_feat_size, img_feat_size, img_att_feat_size ):
        super(MHC3, self).__init__()
        self.proj_i = nn.Linear(img_feat_size, MFB_K * MFB_O)    # __C.MFB_K * __C.MFB_O
        self.proj_q = nn.Linear(ques_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.proj_ia = nn.Linear(img_att_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.proj_qa = nn.Linear(ques_att_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.dropout = nn.Dropout(DROPOUT_R)
        self.pool = nn.AvgPool1d(MFB_K, stride=MFB_K)

        self.linear1 = torch.nn.Sequential(torch.nn.Linear(4 * MFB_O, 10)
                                           #, torch.nn.Softmax()
                                           )
        self.linear2 = torch.nn.Sequential(torch.nn.Linear(MFB_O, 10)
                                           #, torch.nn.Softmax()
                                           )
        self.linear3 = torch.nn.Sequential(torch.nn.Linear(3 * 10, 10)
                                           #, torch.nn.Softmax()
                                           )

    def forward(self, ques_feat, ques_att_feat, img_feat, img_att_feat):
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = torch.tensor(ques_feat, dtype=torch.float32)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)
        ques_att_feat = self.proj_qa(ques_att_feat)
        img_att_feat = self.proj_ia(img_att_feat)

        #第一层
        feat_layer1 = torch.cat([ques_feat, img_feat, ques_att_feat, img_att_feat], dim=2)
        feat_layer1 = self.dropout(feat_layer1)
        feat_layer1 = self.pool(feat_layer1) * MFB_K         # (N, C, O)
        feat_layer1 = torch.sqrt(F.relu(feat_layer1)) - torch.sqrt(F.relu(-feat_layer1))
        feat_layer1 = F.normalize(feat_layer1.view(batch_size, -1))         # (N, C*O)
        feat_layer1 = feat_layer1.view(batch_size, 1, 4 * MFB_O)
        out_layer1 = self.linear1(feat_layer1)

        #第二层
        fus_feat1 = ques_feat * img_feat
        fus_feat2 = ques_att_feat * img_att_feat
        fus_feat3 = ques_feat * ques_att_feat
        fus_feat4 = img_feat * img_att_feat

        feat_layer2 = torch.cat([fus_feat1, fus_feat2, fus_feat3, fus_feat4], dim=2)
        feat_layer2 = self.dropout(feat_layer2)
        feat_layer2 = self.pool(feat_layer2) * MFB_K         # (N, C, O)
        feat_layer2 = torch.sqrt(F.relu(feat_layer2)) - torch.sqrt(F.relu(-feat_layer2))
        feat_layer2 = F.normalize(feat_layer2.view(batch_size, -1))         # (N, C*O)
        feat_layer2 = feat_layer2.view(batch_size, 1, 4 * MFB_O)
        out_layer2 = self.linear1(feat_layer2)

        # 第三层
        all_fus_feat = fus_feat1 * fus_feat2 * fus_feat3 * fus_feat4

        feat_layer3 = self.dropout(all_fus_feat)
        feat_layer3 = self.pool(feat_layer3) * MFB_K  # (N, C, O)
        feat_layer3 = torch.sqrt(F.relu(feat_layer3)) - torch.sqrt(F.relu(-feat_layer3))
        feat_layer3 = F.normalize(feat_layer3.view(batch_size, -1))  # (N, C*O)
        feat_layer3 = feat_layer3.view(batch_size, 1, MFB_O)
        out_layer3 = self.linear2(feat_layer3)

        total_out = torch.cat([out_layer1, out_layer2, out_layer3], dim=2)
        total_out = self.linear3(total_out)

        return out_layer1, out_layer2, out_layer3, total_out

class MHC3_no_cross_modal_fusion_layer(nn.Module):                  #10分类
    def __init__(self, ques_feat_size, ques_att_feat_size, img_feat_size, img_att_feat_size ):
        super(MHC3_no_cross_modal_fusion_layer, self).__init__()
        self.proj_i = nn.Linear(img_feat_size, MFB_K * MFB_O)    # __C.MFB_K * __C.MFB_O
        self.proj_q = nn.Linear(ques_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.proj_ia = nn.Linear(img_att_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.proj_qa = nn.Linear(ques_att_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.dropout = nn.Dropout(DROPOUT_R)
        self.pool = nn.AvgPool1d(MFB_K, stride=MFB_K)

        self.linear1 = torch.nn.Sequential(torch.nn.Linear(4 * MFB_O, 10)
                                           #, torch.nn.Softmax()
                                           )
        self.linear2 = torch.nn.Sequential(torch.nn.Linear(MFB_O, 10)
                                           #, torch.nn.Softmax()
                                           )
        self.linear3 = torch.nn.Sequential(torch.nn.Linear(2 * 10, 10)
                                           #, torch.nn.Softmax()
                                           )

    def forward(self, ques_feat, ques_att_feat, img_feat, img_att_feat):
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = torch.tensor(ques_feat, dtype=torch.float32)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)
        ques_att_feat = self.proj_qa(ques_att_feat)
        img_att_feat = self.proj_ia(img_att_feat)

        #第一层
        feat_layer1 = torch.cat([ques_feat, img_feat, ques_att_feat, img_att_feat], dim=2)
        feat_layer1 = self.dropout(feat_layer1)
        feat_layer1 = self.pool(feat_layer1) * MFB_K         # (N, C, O)
        feat_layer1 = torch.sqrt(F.relu(feat_layer1)) - torch.sqrt(F.relu(-feat_layer1))
        feat_layer1 = F.normalize(feat_layer1.view(batch_size, -1))         # (N, C*O)
        feat_layer1 = feat_layer1.view(batch_size, 1, 4 * MFB_O)
        out_layer1 = self.linear1(feat_layer1)

        #第二层

        all_fus_feat = ques_feat * img_feat * ques_att_feat * img_att_feat

        feat_layer3 = self.dropout(all_fus_feat)
        feat_layer3 = self.pool(feat_layer3) * MFB_K  # (N, C, O)
        feat_layer3 = torch.sqrt(F.relu(feat_layer3)) - torch.sqrt(F.relu(-feat_layer3))
        feat_layer3 = F.normalize(feat_layer3.view(batch_size, -1))  # (N, C*O)
        feat_layer3 = feat_layer3.view(batch_size, 1, MFB_O)
        out_layer3 = self.linear2(feat_layer3)

        total_out = torch.cat([out_layer1, out_layer3], dim=2)
        total_out = self.linear3(total_out)

        return out_layer1, total_out, out_layer3, total_out



class MHC3_no_global_fusion_layer(nn.Module):                  #10分类
    def __init__(self, ques_feat_size, ques_att_feat_size, img_feat_size, img_att_feat_size ):
        super(MHC3_no_global_fusion_layer, self).__init__()
        self.proj_i = nn.Linear(img_feat_size, MFB_K * MFB_O)    # __C.MFB_K * __C.MFB_O
        self.proj_q = nn.Linear(ques_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.proj_ia = nn.Linear(img_att_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.proj_qa = nn.Linear(ques_att_feat_size, MFB_K * MFB_O)   # __C.MFB_K * __C.MFB_O
        self.dropout = nn.Dropout(DROPOUT_R)
        self.pool = nn.AvgPool1d(MFB_K, stride=MFB_K)

        self.linear1 = torch.nn.Sequential(torch.nn.Linear(4 * MFB_O, 10)
                                           #, torch.nn.Softmax()
                                           )
        self.linear2 = torch.nn.Sequential(torch.nn.Linear(MFB_O, 10)
                                           #, torch.nn.Softmax()
                                           )
        self.linear3 = torch.nn.Sequential(torch.nn.Linear(3 * 10, 10)
                                           #, torch.nn.Softmax()
                                           )

    def forward(self, ques_feat, ques_att_feat, img_feat, img_att_feat):
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = torch.tensor(ques_feat, dtype=torch.float32)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)
        ques_att_feat = self.proj_qa(ques_att_feat)
        img_att_feat = self.proj_ia(img_att_feat)

        #第一层
        feat_layer1 = torch.cat([ques_feat, img_feat, ques_att_feat, img_att_feat], dim=2)
        feat_layer1 = self.dropout(feat_layer1)
        feat_layer1 = self.pool(feat_layer1) * MFB_K         # (N, C, O)
        feat_layer1 = torch.sqrt(F.relu(feat_layer1)) - torch.sqrt(F.relu(-feat_layer1))
        feat_layer1 = F.normalize(feat_layer1.view(batch_size, -1))         # (N, C*O)
        feat_layer1 = feat_layer1.view(batch_size, 1, 4 * MFB_O)
        out_layer1 = self.linear1(feat_layer1)

        #第二层
        fus_feat1 = ques_feat * img_feat
        fus_feat2 = ques_att_feat * img_att_feat
        fus_feat3 = ques_feat * ques_att_feat
        fus_feat4 = img_feat * img_att_feat

        feat_layer2 = torch.cat([fus_feat1, fus_feat2, fus_feat3, fus_feat4], dim=2)
        feat_layer2 = self.dropout(feat_layer2)
        feat_layer2 = self.pool(feat_layer2) * MFB_K         # (N, C, O)
        feat_layer2 = torch.sqrt(F.relu(feat_layer2)) - torch.sqrt(F.relu(-feat_layer2))
        feat_layer2 = F.normalize(feat_layer2.view(batch_size, -1))         # (N, C*O)
        feat_layer2 = feat_layer2.view(batch_size, 1, 4 * MFB_O)
        out_layer2 = self.linear1(feat_layer2)

        return out_layer1, out_layer2, out_layer1, out_layer2

class multi_CoAtt_and_Independent2(nn.Module):
    def __init__(self):
        super(multi_CoAtt_and_Independent2, self).__init__()

        img_feat_size = LSTM_OUT_SIZE
        img_att_feat_size = img_feat_size * I_GLIMPSES
        ques_att_feat_size = LSTM_OUT_SIZE * Q_GLIMPSES

        self.zip = nn.Linear(1024, LSTM_OUT_SIZE * Q_GLIMPSES)

        self.q_att = QAtt()
        self.i_att = IAtt(img_feat_size, ques_att_feat_size)

        self.mhc = MHC(ques_att_feat_size, ques_att_feat_size, img_att_feat_size, img_att_feat_size)
        #self.mhc = MHC_no_cross_modal_fusion_layer(ques_att_feat_size, ques_att_feat_size, img_att_feat_size, img_att_feat_size)
        #self.mhc = MHC_no_global_fusion_layer(ques_att_feat_size, ques_att_feat_size, img_att_feat_size, img_att_feat_size)


        self.output1 = torch.nn.Sequential(nn.Linear(3*MFB_O, ouput_mid),
                                torch.nn.ReLU())
        self.output2 = torch.nn.Sequential(nn.Linear(ouput_mid, 3),
                                torch.nn.ReLU())
        self.total_out = nn.Linear(9, 3)




    def forward(self, ques_feat, img_feat, correlation):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        ques_feat_temp = ques_feat.view(ques_feat.size(0), int(1024 / LSTM_OUT_SIZE),
                                   LSTM_OUT_SIZE)  # 更改维度
        img_feat_temp = img_feat.view(img_feat.size(0), int(1024 / FRCN_FEAT_SIZE),
                                 FRCN_FEAT_SIZE)
        ques_att_feat = self.q_att(ques_feat_temp)               # (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        fuse_att_feat = self.i_att(img_feat_temp, ques_att_feat ,correlation)     # (N, FRCN_FEAT_SIZE*I_GLIMPSES)
        ques_feat = self.zip(ques_feat)
        img_feat = self.zip(img_feat)

        out_layer1, out_layer2, out_layer3, total_out = self.mhc(ques_feat, ques_att_feat.unsqueeze(1), img_feat, fuse_att_feat.unsqueeze(1))             # z:(N, 1, O)  _:(N, C, K*O)
        #out_layer1, out_layer2, out_layer3, total_out = self.mhc(ques_feat, ques_feat, img_feat, fuse_att_feat.unsqueeze(1))             # 不用text attention
        #out_layer1, out_layer2, out_layer3, total_out = self.mhc(ques_feat, ques_att_feat.unsqueeze(1), img_feat, img_feat)             # 不用img attention

        out_layer1 = out_layer1.squeeze(1)                                                            # (N, O)
        out_layer2 = out_layer2.squeeze(1)                                                            # (N, O)
        out_layer3 = out_layer3.squeeze(1)                                                            # (N, O)
        total_out = total_out.squeeze(1)                                                            # (N, O)


        return out_layer1, out_layer2, out_layer3, total_out


class multi_CoAtt_and_Independent3(nn.Module):          #ZOL5分类，用MHFF
    def __init__(self):
        super(multi_CoAtt_and_Independent3, self).__init__()

        img_feat_size = LSTM_OUT_SIZE
        img_att_feat_size = img_feat_size * I_GLIMPSES
        ques_att_feat_size = LSTM_OUT_SIZE * Q_GLIMPSES

        self.uniform1 = torch.nn.Sequential(nn.Linear(774, 1024),
                                torch.nn.ReLU())
        #self.uniform1 = nn.Linear(768, 1024)
        self.uniform2 = torch.nn.Sequential(nn.Linear(4096, 1024),
                                torch.nn.ReLU())
        self.zip = torch.nn.Sequential(nn.Linear(1024, LSTM_OUT_SIZE * Q_GLIMPSES),
                                torch.nn.ReLU())


        self.q_att = QAtt()
        self.i_att = IAtt_NOc(img_feat_size, ques_att_feat_size)

        self.mhc = MHC2(ques_att_feat_size, ques_att_feat_size, img_att_feat_size, img_att_feat_size)




    def forward(self, ques_feat, img_feat):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        ques_feat = self.uniform1(ques_feat)
        img_feat = self.uniform2(img_feat)
        ques_feat_temp = ques_feat.view(ques_feat.size(0), int(1024 / LSTM_OUT_SIZE),
                                   LSTM_OUT_SIZE)  # 更改维度
        img_feat_temp = img_feat.view(img_feat.size(0), int(1024 / FRCN_FEAT_SIZE),
                                 FRCN_FEAT_SIZE)
        ques_att_feat = self.q_att(ques_feat_temp)               # (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        fuse_att_feat = self.i_att(img_feat_temp, ques_att_feat)     # (N, FRCN_FEAT_SIZE*I_GLIMPSES)
        ques_feat = self.zip(ques_feat)
        img_feat = self.zip(img_feat)

        out_layer1, out_layer2, out_layer3, total_out = self.mhc(ques_feat, ques_att_feat.unsqueeze(1), img_feat, fuse_att_feat.unsqueeze(1))             # z:(N, 1, O)  _:(N, C, K*O)
        #out_layer1, out_layer2, out_layer3, total_out = self.mhc(ques_feat, ques_feat, img_feat, fuse_att_feat.unsqueeze(1))             # 不用text attention
        #out_layer1, out_layer2, out_layer3, total_out = self.mhc(ques_feat, ques_att_feat.unsqueeze(1), img_feat, img_feat)             # 不用img attention

        out_layer1 = out_layer1.squeeze(1)                                                            # (N, O)
        out_layer2 = out_layer2.squeeze(1)                                                            # (N, O)
        out_layer3 = out_layer3.squeeze(1)                                                            # (N, O)
        total_out = total_out.squeeze(1)                                                            # (N, O)


        return out_layer1, out_layer2, out_layer3, total_out



class CoAtt_and_Independent_ZOL(nn.Module):             #ZOL5分类，不用MHFF
    def __init__(self):
        super(CoAtt_and_Independent_ZOL, self).__init__()

        img_feat_size = LSTM_OUT_SIZE
        img_att_feat_size = img_feat_size * I_GLIMPSES
        ques_att_feat_size = LSTM_OUT_SIZE * Q_GLIMPSES
        self.zip = nn.Linear(1024, MFB_O)
        self.uniform1 = torch.nn.Sequential(nn.Linear(774, 1024),
                                torch.nn.ReLU())
        #self.uniform1 = nn.Linear(768, 1024)
        self.uniform2 = torch.nn.Sequential(nn.Linear(4096, 1024),
                                torch.nn.ReLU())
        self.q_att = QAtt()
        self.i_att = IAtt_NOc(img_feat_size, ques_att_feat_size)

        self.mfb = MFB(img_att_feat_size, ques_att_feat_size)
        self.output1 = torch.nn.Sequential(nn.Linear(3 * MFB_O, 5),
                                torch.nn.ReLU())
        #self.output1 = torch.nn.Sequential(nn.Linear(MFB_O, ouput_mid),
        #                                   torch.nn.ReLU())
        #self.output2 = torch.nn.Sequential(nn.Linear(ouput_mid, 3),
        #                        torch.nn.ReLU())
        self.total_out = nn.Linear(15, 5)
        self.total_out2 = nn.Linear(10, 5)      # w/o MHFF + text-att + img-att





    def forward(self, ques_feat, text_, img_feat, img_):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        model1 =RNNnet_ZOL()
        model2 =CNNnet_ZOL()
        ques_feat = self.uniform1(ques_feat)
        text_ = self.uniform1(text_)
        img_feat = self.uniform2(img_feat)
        img_ = self.uniform2(img_)
        text = model1(text_)
        img_feature = img_.reshape(img_.size(0), 1, 32, 32)
        img = model2(img_feature)
        text2 = self.zip(ques_feat).squeeze(1)
        img2 = self.zip(img_feat).squeeze(1)
        ques_feat = ques_feat.view(ques_feat.size(0), int(1024 / LSTM_OUT_SIZE),
                                   LSTM_OUT_SIZE)  # 更改维度
        img_feat = img_feat.view(img_feat.size(0), int(1024 / FRCN_FEAT_SIZE),
                                 FRCN_FEAT_SIZE)
        ques_feat = self.q_att(ques_feat)               # (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        fuse_feat = self.i_att(img_feat, ques_feat)     # (N, FRCN_FEAT_SIZE*I_GLIMPSES)


        z, _ = self.mfb(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))             # z:(N, 1, O)  _:(N, C, K*O)
        z = z.squeeze(1)                                                            # (N, O)
        z = torch.cat([z,text2,img2],dim=1)
        z = self.output1(z)
        #z = self.output2(z)


        total = torch.cat([text,img],dim=1)
        total = torch.cat([total,z],dim=1)
        total = self.total_out(total)

        """
        # w/o MHFF + text-att + img-att

        model1 =RNNnet_ZOL()
        model2 =CNNnet_ZOL()

        text_ = self.uniform1(text_)
        img_ = self.uniform2(img_)


        text = model1(text_)
        img_feature = img_.reshape(img_.size(0), 1, 32, 32)
        img = model2(img_feature)

        total = torch.cat([text, img], dim=1)
        total = self.total_out2(total)
        """

        return total


class multi_CoAtt_and_Independent4(nn.Module):          #ZOL10分类，用MHFF
    def __init__(self):
        super(multi_CoAtt_and_Independent4, self).__init__()

        img_feat_size = LSTM_OUT_SIZE
        img_att_feat_size = img_feat_size * I_GLIMPSES
        ques_att_feat_size = LSTM_OUT_SIZE * Q_GLIMPSES

        self.uniform1 = torch.nn.Sequential(nn.Linear(774, 1024),
                                torch.nn.ReLU())
        #self.uniform1 = nn.Linear(768, 1024)
        self.uniform2 = torch.nn.Sequential(nn.Linear(4096, 1024),
                                torch.nn.ReLU())
        self.zip = torch.nn.Sequential(nn.Linear(1024, LSTM_OUT_SIZE * Q_GLIMPSES),
                                torch.nn.ReLU())


        self.q_att = QAtt()
        self.i_att = IAtt_NOc(img_feat_size, ques_att_feat_size)
        self.i_att_c = IAtt(img_feat_size, ques_att_feat_size)

        #self.mhc = MHC3(ques_att_feat_size, ques_att_feat_size, img_att_feat_size, img_att_feat_size)
        #self.mhc = MHC3_no_cross_modal_fusion_layer(ques_att_feat_size, ques_att_feat_size, img_att_feat_size, img_att_feat_size)
        self.mhc = MHC3_no_global_fusion_layer(ques_att_feat_size, ques_att_feat_size, img_att_feat_size, img_att_feat_size)




    def forward(self, ques_feat, img_feat, correlation=None):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        ques_feat = self.uniform1(ques_feat)
        img_feat = self.uniform2(img_feat)
        ques_feat_temp = ques_feat.view(ques_feat.size(0), int(1024 / LSTM_OUT_SIZE),
                                   LSTM_OUT_SIZE)  # 更改维度
        img_feat_temp = img_feat.view(img_feat.size(0), int(1024 / FRCN_FEAT_SIZE),
                                 FRCN_FEAT_SIZE)
        ques_att_feat = self.q_att(ques_feat_temp)               # (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        if correlation is None:
            fuse_att_feat = self.i_att(img_feat_temp, ques_att_feat)     # (N, FRCN_FEAT_SIZE*I_GLIMPSES)
        else:
            fuse_att_feat = self.i_att_c(img_feat_temp, ques_att_feat, correlation)  # (N, FRCN_FEAT_SIZE*I_GLIMPSES)

        ques_feat = self.zip(ques_feat)
        img_feat = self.zip(img_feat)

        out_layer1, out_layer2, out_layer3, total_out = self.mhc(ques_feat, ques_att_feat.unsqueeze(1), img_feat, fuse_att_feat.unsqueeze(1))             # z:(N, 1, O)  _:(N, C, K*O)
        #out_layer1, out_layer2, out_layer3, total_out = self.mhc(ques_feat, ques_feat, img_feat, fuse_att_feat.unsqueeze(1))             # 不用text attention
        #out_layer1, out_layer2, out_layer3, total_out = self.mhc(ques_feat, ques_att_feat.unsqueeze(1), img_feat, img_feat)             # 不用img attention

        out_layer1 = out_layer1.squeeze(1)                                                            # (N, O)
        out_layer2 = out_layer2.squeeze(1)                                                            # (N, O)
        out_layer3 = out_layer3.squeeze(1)                                                            # (N, O)
        total_out = total_out.squeeze(1)                                                            # (N, O)


        return out_layer1, out_layer2, out_layer3, total_out



class CoAtt_and_Independent_ZOL2(nn.Module):             #ZOL10分类，不用MHFF
    def __init__(self):
        super(CoAtt_and_Independent_ZOL2, self).__init__()

        img_feat_size = LSTM_OUT_SIZE
        img_att_feat_size = img_feat_size * I_GLIMPSES
        ques_att_feat_size = LSTM_OUT_SIZE * Q_GLIMPSES
        self.zip = nn.Linear(1024, MFB_O)
        self.uniform1 = torch.nn.Sequential(nn.Linear(774, 1024),
                                torch.nn.ReLU())
        #self.uniform1 = nn.Linear(768, 1024)
        self.uniform2 = torch.nn.Sequential(nn.Linear(4096, 1024),
                                torch.nn.ReLU())
        self.q_att = QAtt()
        self.i_att = IAtt_NOc(img_feat_size, ques_att_feat_size)

        self.mfb = MFB(img_att_feat_size, ques_att_feat_size)
        self.output1 = torch.nn.Sequential(nn.Linear(3 * MFB_O, 10),
                                torch.nn.ReLU())
        #self.output1 = torch.nn.Sequential(nn.Linear(MFB_O, ouput_mid),
        #                                   torch.nn.ReLU())
        #self.output2 = torch.nn.Sequential(nn.Linear(ouput_mid, 3),
        #                        torch.nn.ReLU())
        self.total_out = nn.Linear(30, 10)
        self.total_out2 = nn.Linear(20, 10)      # w/o MHFF + text-att + img-att





    def forward(self, ques_feat, text_, img_feat, img_):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        model1 =RNNnet_ZOL_10_only_text2()
        model2 =CNNnet_ZOL_10_onlyImg2()
        ques_feat = self.uniform1(ques_feat)
        text_ = self.uniform1(text_)
        img_feat = self.uniform2(img_feat)
        img_ = self.uniform2(img_)
        text = model1(text_)
        img_feature = img_.reshape(img_.size(0), 1, 32, 32)
        img = model2(img_feature)
        text2 = self.zip(ques_feat).squeeze(1)
        img2 = self.zip(img_feat).squeeze(1)
        ques_feat = ques_feat.view(ques_feat.size(0), int(1024 / LSTM_OUT_SIZE),
                                   LSTM_OUT_SIZE)  # 更改维度
        img_feat = img_feat.view(img_feat.size(0), int(1024 / FRCN_FEAT_SIZE),
                                 FRCN_FEAT_SIZE)
        ques_feat = self.q_att(ques_feat)               # (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        fuse_feat = self.i_att(img_feat, ques_feat)     # (N, FRCN_FEAT_SIZE*I_GLIMPSES)


        z, _ = self.mfb(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))             # z:(N, 1, O)  _:(N, C, K*O)
        z = z.squeeze(1)                                                            # (N, O)
        z = torch.cat([z,text2,img2],dim=1)
        z = self.output1(z)
        #z = self.output2(z)


        total = torch.cat([text,img],dim=1)
        total = torch.cat([total,z],dim=1)
        total = self.total_out(total)

        return total