# Code for "Universal high spatial resolution hyperspectral imaging using hybrid-resolution image fusion"
# Author: Xuheng Cao

import cv2
import numpy as np
from torch import nn
import torch
from sklearn import preprocessing,cluster

class subBPNN(nn.Module):
    def __init__(self,msiband=3):
        super(subBPNN, self).__init__()
        self.act = nn.Sigmoid()
        self.layer1 = nn.Linear(msiband,21)
        self.layer2 = nn.Linear(21,msiband)
    def forward(self,x):
        return self.layer2(self.act(self.layer1(x)))


class CBPNN():

    def __init__(self,HSI,RGB,CMF):
        '''
        :param HSI: low spatial resolution hyperspectral image shape:[h,w,c]
        :param RGB: high spatial resolution RGB image shape:[H,W,3]
        :param CMF: CIE Color matching function
        '''

        H,W,c = RGB.shape
        h,w,C = HSI.shape
        self.XYZ = HSI @ CMF
        self.RGB = RGB
        self.msiband = 3
        self.sf = H//h
        self.h,self.w=h,w
    def _mx2tensor(self,data):
        return torch.FloatTensor(data).unsqueeze(0).cuda()
    def _tensor2mx(self,data):

        return data[0].detach().cpu().numpy()
    def get_data(self):
        self.LRMSIData,self.MSIData,self.HSIData=[],[],[]
        self.subnet=[]
        self.location=[]
        LRMSI = cv2.resize(self.RGB, [self.h, self.w])


        XYZ2D = self.XYZ.reshape([-1, self.msiband])
        RGB2D = self.RGB.reshape([-1, self.msiband])
        LRRGB2D = LRMSI.reshape([-1,self.msiband])

        RGB2DP = preprocessing.normalize(RGB2D)
        LRRGB2DP = preprocessing.normalize(LRRGB2D)

        bandwidth = cluster.estimate_bandwidth(LRRGB2DP, quantile=0.2)
        Kcluster = cluster.MeanShift(bandwidth=bandwidth)

        LRMSIidx = Kcluster.fit_predict(LRRGB2DP)
        clust_nums=max(LRMSIidx)

        MSIidx = Kcluster.predict(RGB2DP)
        for clust_num in range(clust_nums+1):
            self.LRMSIData.append(LRRGB2D[LRMSIidx==clust_num])
            self.HSIData.append(XYZ2D[LRMSIidx==clust_num])
            self.MSIData.append(RGB2D[MSIidx==clust_num])
            self.location.append(np.argwhere(MSIidx == clust_num))
            self.subnet.append(subBPNN(self.msiband))

    def poly_expand(self,img2d):
        poly_expand = np.empty([img2d.shape[0],9]).T
        r,g,b = [img2d[:,color_channel] for color_channel in range(3)]
        r2 = pow(r,2)
        g2 = pow(g,2)
        b2 = pow(b,2)
        poly_expand [:3] = [r,g,b]
        poly_expand [3:6] = [r*g,g*b,b*r]
        poly_expand [6:] = [r2,g2,b2]

        return poly_expand.T


    def train_predict(self,max_epoch=5000):
        '''
        :param max_epoch: Max epoch for iteration
        :return: high spatial resolution XYZ image
        '''

        HRXYZ2D = np.empty([self.h*self.w*self.sf*self.sf,self.msiband])
        # Test_ = np.empty([512*512,3])
        Loss = nn.L1Loss()

        for clust_num,mapper in enumerate(self.subnet):

            lrmsi_data = self._mx2tensor(self.LRMSIData[clust_num])
            hsi_data = self._mx2tensor(self.HSIData[clust_num])
            msi_data = self._mx2tensor(self.MSIData[clust_num])
            if msi_data.shape[1] ==0:
                continue

            b,n,c= lrmsi_data.shape
            n_test = int(0.15*n)
            if n_test<4:
                TESTOFF=True
                train_data = lrmsi_data
                train_data_l = hsi_data
            else:
                TESTOFF=False
                train_data = lrmsi_data[:,n_test:]
                train_data_l = hsi_data[:,n_test:]
                test_data = lrmsi_data[:,:n_test]
                test_data_l = hsi_data[:,:n_test]

            mapper.cuda()


            trainer = torch.optim.Adam(params=mapper.parameters(),lr=5e-3)
            sched = torch.optim.lr_scheduler.StepLR(trainer,500,0.95)
            min_loss=1
            test_lossset=[]
            for epoch in range(max_epoch):
                trainer.zero_grad()
                pre_hsi = mapper(train_data)
                loss = Loss(pre_hsi,train_data_l)
                loss.backward()
                trainer.step()
                sched.step()

                if TESTOFF is False and epoch %10==0:
                    mapper.eval()
                    with torch.no_grad():
                        test_pre_hsi = mapper(test_data)
                        test_loss = Loss(test_pre_hsi,test_data_l).item()
                        test_lossset.append(test_loss)
                        if test_loss<min_loss:
                            min_loss = test_loss
                            torch.save(mapper,f'Models/SUBNET_{clust_num}.pth')
                            print(f'\rLearning the {clust_num}th mapping function\t {min_loss}',end='')
                    mapper.train()
            try:
                mapper = torch.load(f'Models/SUBNET_{clust_num}.pth')
            except:
                pass
            recon= mapper(msi_data)
            HRXYZ2D[self.location[clust_num]]= self._tensor2mx(recon)[:, np.newaxis, :]

        return HRXYZ2D.reshape([self.h*self.sf,self.w*self.sf,-1])


