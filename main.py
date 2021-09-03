#!/usr/bin/env python
# coding: utf-8



import os
import pandas as pd
import numpy as np

from scipy.spatial import distance as dist


import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm

from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter
summaryWriter = SummaryWriter("logs/log3")



### preprocessing data

ic50_data=pd.read_excel('final_2019_all_info.xlsx')
ic50_data=ic50_data[['pdbid','affinity(log)']]

### prepare inputs from xyz file


def get_lig_data(xyzfile):
    with open(xyzfile) as lig_f:
        lig_c = lig_f.readlines()
        lig_o = [
            i.split() for i in lig_c
            if i.startswith(('N ', 'C ', 'O ', 'Cl ', 'F ', 'S '))  #leave out H
        ]
        return lig_o


class mol_data():
    def __init__(self, pdbid):
        self.pdbid = pdbid
        self.path = '/data/mdml/Data/pt-2510-model/pocket-ligand-all/'
        self.lig_name = self.path + 'ligand-xyz/' + pdbid + '_ligand.xyz'
        self.poc_name = self.path + 'pocket-xyz/' + pdbid + '_pocket.xyz'
        self.lig = get_lig_data(self.lig_name)
        self.poc = get_lig_data(self.poc_name)
#         self.com = self.lig
        self.com = self.lig + self.poc
        self.atom_list = {
            'N': 1,
            'C': 2,
            'O': 3,
            'Cl': 4,
            'F': 5,
            'S': 6,
            'X': 0
        }


#         self.z_size = 800

    def get_pos(self):
        pos = [[float(i[1]), float(i[2]), float(i[3])] for i in self.com]
        pos = np.array(pos)
        return pos

    def get_elem(self):
        elem = [i[0] for i in self.com]
        return elem

    def get_elem_label(self):
        elem_label = [self.atom_list[i[0]] for i in self.com]
        #         padding = [0 for i in range(self.z_size - len(elem_label))]
        #         elem_label = elem_label + padding
        return elem_label

    def get_num(self):
        num = [i for i in range(len(self.com))]
        return num

    def dist_ed(self, A):
        length = A.shape[0]
        dist_e = np.zeros((length, length))
        for i in range(length):
            for j in range(length):
                dist_e[i][j] = dist.euclidean(A[i], A[j])
        return dist_e

    def dist_ed_partial(self, A, lig_num=None):
        length = A.shape[0]
        dist_e = np.zeros((length, length))
        for i in range(length):
            for j in range(length):
#                 if (i-lig_num+1)*(j-lig_num+1)<0:
                    dist_e[i][j] = dist.euclidean(A[i], A[j])
        return dist_e
    

    def mol_to_presentation(self, Z=None, POS=None, cut_down=0, cut_up=4):
        Z = self.get_elem_label()
        atom_num = len(Z)
        POS = self.get_pos()
#         idx_i_ldist_e np.arange(len(POS))  # atom label from 1
#         dist_e = self.dist_ed(POS)
        dist_e = self.dist_ed_partial(POS,lig_num=len(self.lig))

        #         ij_label=np.argwhere(dist_e<cutoff )
        ij_label = np.argwhere(
            np.logical_and(dist_e > cut_down, dist_e < cut_up))
        ij_order = [i for i in range(len(ij_label))]

        #     np.where(np.all(ij_label == [47, 46], axis=1))[0]   example: search based on idx_i and idx_j
        #     np.argmax(ij_label==[47, 46])   example: search based on idx_i and idx_j

        idx_i = ij_label[:, 0]  # start atom
        idx_j = ij_label[:, 1]  # end atom
        dist_ij = [dist_e[idx_i[i], idx_j[i]] for i in range(len(idx_i))]
        dist_ij = np.array(dist_ij)
        #         idx_i_j2_num = [len(idx_i[idx_i == i]) for i in range(atom_num)
        #                         ]  # number of all atoms j near i within cutoff
        #         idx_i_j2 = [idx_j[idx_i == i] for i in range(atom_num)
        #                     ]  # all atoms j  near i within cutoff

        #         idx_i_j_j2_i = [
        #             np.repeat(idx_i[i], idx_i_j2_num[idx_i[i]])
        #             for i in range(len(idx_i))
        #         ]
        #         idx_i_j_j2_j = [
        #             np.repeat(idx_j[i], idx_i_j2_num[idx_i[i]])
        #             for i in range(len(idx_i))
        #         ]
        #         idx_i_j_j2_j2 = [
        #             list(idx_i_j2[i]) * idx_i_j2_num[i] for i in range(atom_num)
        #         ]

        #         idx_i_j_j2_i = np.array([i for aaa in idx_i_j_j2_i for i in aaa])
        #         idx_i_j_j2_j = np.array([i for aaa in idx_i_j_j2_j for i in aaa])
        #         idx_i_j_j2_j2 = np.array([i for aaa in idx_i_j_j2_j2 for i in aaa])

        #         mask = idx_i_j_j2_j != idx_i_j_j2_j2
        #         idx_i_j_j2_i = idx_i_j_j2_i[mask]
        #         idx_i_j_j2_j = idx_i_j_j2_j[mask]
        #         idx_i_j_j2_j2 = idx_i_j_j2_j2[mask]

        dist_e = torch.tensor(dist_e)
        Z = torch.tensor(Z)
        idx_i = torch.tensor(idx_i)
        idx_j = torch.tensor(idx_j)
        dist_ij = torch.tensor(dist_ij)
        #         idx_i_j_j2_i = torch.tensor(idx_i_j_j2_i)
        #         idx_i_j_j2_j = torch.tensor(idx_i_j_j2_j)
        #         idx_i_j_j2_j2 = torch.tensor(idx_i_j_j2_j2)
        #         POS = torch.tensor(POS)
        ic50 = torch.tensor(ic50_data[ic50_data['pdbid'] == self.pdbid]
                            ['affinity(log)'].values)

        #         return self.pdbid, ic50, Z, POS, idx_i, idx_j, dist_ij, idx_i_j_j2_i, idx_i_j_j2_j, idx_i_j_j2_j2
        return self.pdbid, len(self.lig), ic50, Z, POS, idx_i, idx_j, dist_ij


### train models


class HorseNet(nn.Module):
    def __init__(self):
        super(HorseNet, self).__init__()
        self.z_length = 1000
        self.n_iblock = 3
        #hyper part
        self.embedding_hz1_in = 7
        self.embedding_hz1_out = 64

        

        #cfconv
        self.Cfconv_hz1_in = 1
        self.Cfconv_hz1_out = 64
        self.Cfconv_hz2_in = 64
        self.Cfconv_hz2_out = 64

        #iblock
        self.Iblock_hz1_in = 64
        self.Iblock_hz1_out = 64
        self.Iblock_hz2_in = 64
        self.Iblock_hz2_out = 64
        self.Iblock_hz3_in = 64
        self.Iblock_hz3_out = 64

        #horsenet
        self.net_hz0_in=64
        self.net_hz0_out=64
        self.net_hz1_in = 64
        self.net_hz1_out = 64
        self.net_hz2_in = 64
        self.net_hz2_out = 1

        #embedding
        self.embeds = nn.Embedding(self.embedding_hz1_in,self.embedding_hz1_out)

        #cfconvmodule
        self.Cfconv_l1 = nn.Linear(self.Cfconv_hz1_in, self.Cfconv_hz1_out)
        self.Cfconv_l2 = nn.Linear(self.Cfconv_hz2_in, self.Cfconv_hz2_out)

        #iblockmodule
        self.Iblock_l1 = nn.Linear(self.Iblock_hz1_in, self.Iblock_hz1_out)
        self.Iblock_l2 = nn.Linear(self.Iblock_hz2_in, self.Iblock_hz2_out)
        self.Iblock_l3 = nn.Linear(self.Iblock_hz3_in, self.Iblock_hz3_out)

        #horsenetmodule
        self.net_l0 = nn.Linear(self.net_hz0_in, self.net_hz0_out)
        self.net_l1 = nn.Linear(self.net_hz1_in, self.net_hz1_out)
        self.net_l2 = nn.Linear(self.net_hz2_in, self.net_hz2_out)
        
    def poolseg(self,x,segs):
        num_idx = segs[-1] + 1
        y=[]
        # y = torch.zeros(num_idx, requires_grad=True)
        for i in range(num_idx):
            y0=x[segs==i].mean()
            y.append(y0)
        # y=torch.tensor(y,requires_grad=True)
        return torch.stack(y)


    def HorseEmbed(self, Z):
        x = self.embeds(Z)
        return x

    def HorseCfconv(self, x, idx_i, idx_j, dist_ij):
        dist_ij = dist_ij.view(-1, 1).float()
        rbf = F.leaky_relu(self.Cfconv_l1(dist_ij), 0.1)
        rbf = F.leaky_relu(self.Cfconv_l2(rbf), 0.1)
        j_feature = torch.index_select(x, 0, idx_j)
        j_new = torch.mul(j_feature, rbf)
        for i in range(len(x)):
            mask = idx_i == i
            j_new_mask = j_new[mask]
            x[i] = torch.sum(j_new_mask, axis=0)
        return x

    def HorseIblock(self, x, idx_i, idx_j, dist_ij):
        x_new = self.Iblock_l1(x)
        x_new = self.HorseCfconv(x_new, idx_i, idx_j, dist_ij)  ##TODO
        x_new = F.leaky_relu(self.Iblock_l2(x_new))
        x_new = self.Iblock_l3(x_new)
        x_final = x + x_new
        return x_final

#     def forward(self, Z, POS, idx_i, idx_j, dist_ij, idx_i_j_j2_i, idx_i_j_j2_j, idx_i_j_j2_j2):
    def forward(self, inputs, batch_size=8):
        segs, Z, POS, idx_i, idx_j, dist_ij=inputs      
        x = self.HorseEmbed(Z)
        x = F.leaky_relu(self.net_l0(x),0.1)
#         for i in range(self.n_iblock):
        for i in range(1):
            x = self.HorseIblock(x, idx_i, idx_j, dist_ij)

        x = F.leaky_relu(self.net_l1(x),0.1)
        x = F.leaky_relu(self.net_l2(x),0.1)
        x = self.poolseg(x,segs)
        return x

def get_atom_indices(data_origin,batch_size):
    data_all=[]
    for k in range(int(len(data_origin)/batch_size)):
        data0=data_origin[k*batch_size:(k+1)*batch_size]
        segs=[]
        num_atoms=[len(data0[i][3]) for i in range(batch_size)]
        num_atoms=[0]+num_atoms
        nums=0
        for j in range(batch_size):
            nums=nums+num_atoms[j]
            seg_m=torch.tensor(np.repeat(j,len(data0[j][3])))
            data0[j][5]=data0[j][5]+nums
            data0[j][6]=data0[j][6]+nums
            segs.append(seg_m)
        
#         ic50, Z, POS, idx_i, idx_j, dist_ij, idx_i_j_j2_i, idx_i_j_j2_j, idx_i_j_j2_j2=[np.concatenate(data0[:,i],axis=0) for i in range(1,10)]
        ic50, Z, POS, idx_i, idx_j, dist_ij=[np.concatenate(data0[:,i],axis=0) for i in range(2,8)]
        lig_num=data0[:,1]
        lig_num=lig_num.astype(int) 
        lig_num=torch.tensor(lig_num)
        pdbid=data0[:,0].tolist()
        segs=np.concatenate(np.array(segs),axis=0)
#         data_new=pdbid, ic50, segs, Z, POS, idx_i, idx_j, dist_ij, idx_i_j_j2_i, idx_i_j_j2_j, idx_i_j_j2_j2
        data_new=pdbid, ic50, lig_num, segs, Z, POS, idx_i, idx_j, dist_ij
                                                                                                                                                                                                   
        data_new=[c if type(c) != np.ndarray else torch.tensor(c) for c in data_new]
        data_all.append(data_new)  
    return data_all



### load data


com=np.load('repres-25-4-com2.npy',allow_pickle=True)
pro=np.load('repres-25-4-protein.npy',allow_pickle=True)
lig=np.load('repres-25-4-ligand.npy',allow_pickle=True)

train_data=get_atom_indices(com, 8)
train_data_pro=get_atom_indices(pro, 8)
train_data_lig=get_atom_indices(lig, 8)

### training model

criterion = nn.MSELoss(reduction='mean')
model=HorseNet().cuda()
optimizer=optim.Adam(model.parameters(), lr=1e-3)


for  epoch in tqdm(range(100)):
    loss=0
    for i in tqdm(range(len(train_data))):
        input=train_data[i][3:9]
        input_pro=train_data_pro[i][3:9]
        input_lig=train_data_lig[i][3:9]        
        input=[xx.cuda() for xx in input]
        input_pro=[xx.cuda() for xx in input_pro]
        input_lig=[xx.cuda() for xx in input_lig]
        target=train_data[i][1].float()
        target=target.cuda()
        optimizer.zero_grad()
        score=model(input,batch_size=8)-model(input_pro,batch_size=8)-model(input_lig,batch_size=8)
        loss_0=criterion(score,target)
        loss_0.backward(retain_graph=False)
        optimizer.step()
        loss=loss+loss_0
        print(i, loss_0)
        average_loss=loss/i
        summaryWriter.add_scalars("loss_batch", {"loss_batch": loss_0}, i)
    print(epoch, average_loss)
    summaryWriter.add_scalars("loss_epoch", {"loss_epoch": average_loss}, epoch)
    

print('ok')

torch.save(model, './model_all.pt')


if __name__='__main__':
      torch.load('model_dict.pt')
      all_data=np.load('select-40.npy',allow_pickle=True)

      train_ratio = 0.75
      validation_ratio = 0.15
      test_ratio = 0.10

      # train is now 75% of the entire data set
      # the _junk suffix means that we drop that variable completely
      x_train, x_test = train_test_split(all_data, test_size=1 - train_ratio)

      # test is now 10% of the initial data set
      # validation is now 15% of the initial data set
      x_val, x_test = train_test_split(x_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

      print(x_train, x_val, x_test)

