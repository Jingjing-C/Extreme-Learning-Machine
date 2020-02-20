# -*- coding: utf-8 -*-
import scipy.io as scio
from numpy import *
import numpy as np
import time
import math
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.datasets import load_iris  #数据集
from sklearn.model_selection import train_test_split  #数据集的分割函数
from sklearn.preprocessing import StandardScaler      #数据预处理
from sklearn import metrics
from sklearn.model_selection import cross_validate




class Traditional_ELM:
    def __init__(self,train_x,train_y,test_x,test_y,NumHiddenNotes):
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.NumHiddenNotes = NumHiddenNotes  #隐层节点数（人工设置）
        self.n_feature = train_x.shape[1]    #列数
        self.n_row = train_x.shape[0]#shape[0]计算行数，shape【1】计算列
        self.factor = np.empty(self.n_row)
        self.b_row = len(test_y)
        self.runtime = time.time()
    
#------InputWeight,输入和隐层之间的权重矩阵；bias,隐含层节点偏置-----------       
    def Parameter(self):
        #a/输入权重
        self.InputWeight = (2*np.random.random([self.n_feature,self.NumHiddenNotes])-1)    #生成shape为（self.n_feature,self.NumHiddenNotes）的数组
        #b/随机产生得输入偏置
        self.Bias = np.random.random((self.n_row,self.NumHiddenNotes))    
        #self.factor = np.dot(self.train_x,self.InputWeight)+ self.Bias #wx+b
        en_one = OneHotEncoder()
        self.train_y = en_one.fit_transform(self.train_y.reshape(-1,1)).toarray() #数据标签需独热编码
        self.test_y = en_one.fit_transform(self.test_y.reshape(-1,1)).toarray()
        print('----------w-------------:\n ',self.InputWeight.shape)
        print('----------b-------------:\n ',self.Bias.shape)
        print('----------train_y----------:\n ',self.train_y.shape)
        print('----------test_y-----------:\n ',self.test_y.shape)

        
#------Activate_function---------------------------------------------------
    def Activation(self,type,Factor):     #Factor为输入训练样本
#        self.factor = np.dot(Factor,self.InputWeight)+ self.Bias     #计算ax+b
        if(type == 'sigmoid'):
#            self.H = 1/(1+np.exp(-self.factor))
            self.H = 1/(1+np.exp(-(np.dot(Factor,self.InputWeight)+ self.Bias)))
            
            print('---------H.shape---------:\n',self.H.shape)
        if(type == 'tanh'):
            self.H = (np.exp(self.factor)-np.exp(-self.factor))/(np.exp(self.factor)+np.exp(-self.factor))
        if(type == 'Fourier'):
            self.H = np.sin(self.factor)
        if(type == 'relu'):
            rnd = self.factor>0
            self.H = self.factor
        
        return self.H
        #print('the shape of factor is:  ',self.factor.shape)
        
#---------------------weight------------------------------------------------
    def weight_(self,train_y,n,p):
        
        self.Weight = np.eye(self.n_row,self.n_row)
        for k in range(self.n_row):
            if train_y[k] == 1:
                self.Weight[k][k] = float(1/p)
            else:
                self.Weight[k][k] = float(1/n)

      
       
    
#----Train ELM------------------------------------------------------------
    def Train_ELM(self,type):
        I_T = np.eye(int(self.n_row), dtype=int)
        C = 2**22
        if(type == 'none'):
            #训练集的输出权重，即beta
#            self.OutputWeight = np.dot(np.dot((mat(I_T/C + np.dot(self.H.T,self.H))).I,self.H.T),self.train_y)#计算输出权重
         #加权重weight
            self.OutputWeight = np.dot(np.dot(np.dot(np.linalg.pinv(I_T/C + np.dot(np.dot(self.H.T,self.Weight),self.H)),self.H.T),self.Weight),self.train_y)
            
        if(type == 'lp'):
  #          self.OutputWeight = np.dot(np.dot(self.H.T,np.linalg.pinv(np.dot(self.H,self.H.T))),self.train_y)
            
#            self.OutputWeight = np.dot(np.dot(self.H.T,np.linalg.pinv(I_T/C + np.dot(self.H,self.H.T))),self.train_y)
#加权重weight
            self.OutputWeight = np.dot(np.dot(np.dot(self.H.T,np.linalg.pinv(I_T/C + np.dot(np.dot(self.Weight,self.H),self.H.T))),self.Weight),self.train_y)
            print('betashape:\n',self.OutputWeight.shape)
        self.Train_elm_predict = np.dot(self.H,self.OutputWeight)
        self.runtime = time.time()-self.runtime
        
        return self.Train_elm_predict
        
        
        
#-----Test ELM,用测试数据训练elm--------------------------------------------
    def Test_ELM(self): #type是激活函数的类型
#        Bias_test = np.random.random(((self.test_x.shape[0],self.NumHiddenNotes)))
#        factor_test = np.dot(self.test_x,self.InputWeight)+Bias_test
#        Htest = 1/(1+np.exp(factor_test))
        print('self.b_row',self.b_row)
        print('self.test_x',self.test_x.shape)
        print('self.InputWeight[:self.b_row,:]',self.InputWeight[:self.b_row,:].shape)
        print('self.Bias[:self.b_row,:]',self.Bias[:self.b_row,:].shape)

##-------------------------------------------------
#       ##只有welm时的输出
#        Htest = 1/(1+np.exp(-(np.dot(self.test_x,self.InputWeight[:self.b_row,:])+ self.Bias[:self.b_row,:])))
#        self.Test_elm_predict = np.dot(Htest,self.OutputWeight) 
#        return self.Test_elm_predict
#       ##########
##-------------------------------------------------
#-------------for tradaboost-------------
        m = self.test_x.shape[0] - self.Bias[:self.b_row,:].shape[0]
        n = self.Bias[:self.b_row,:].shape[1]
        nn = np.zeros((m,n))
        
        B = np.concatenate((self.Bias[:self.b_row,:],nn),axis=0)
        print('~~~~~B',B)
        
        
        Htest = 1/(1+np.exp(-(np.dot(self.test_x,self.InputWeight[:self.b_row,:])+ B)))
#------------------------------------------------------------------------------------------
#
        print('Htest',Htest.shape)
        self.Test_elm_predict = np.dot(Htest,self.OutputWeight)
        
        NewIndex_test_elm_predict = np.argmax(self.Test_elm_predict,1)
#        self.Test_elm_predict = np.dot(Htest[:self.b_row,:],self.OutputWeight)#这里Actural_y表示elm分类实际输出
#        print('Htest[:self.b_row,:]shape:\n',Htest[:self.b_row,:].shape)
        print('NewIndex_test_elm_predict',NewIndex_test_elm_predict)
        
        return NewIndex_test_elm_predict
        
#-----calculate-traindata-accuracy--------------------------------------------
    def Accuracy_Train_ELM(self,type):  #type指激活函数类型
        
        NewIndex = np.argmax(self.Train_elm_predict, 1)+1
        OriIndex = np.argmax(self.train_y,1)+1
        ind = np.where(NewIndex==OriIndex)
        self.TrainAccuracy = np.floor(len(ind[0]))/len(self.train_y)
        

#-----calculate-testdata-accuracy--------------------------------------------       
    def Accuracy_Test_ELM(self,type):
        
        NewIndex_test = np.argmax(self.Test_elm_predict,1)+1
        OriIndex_test = np.argmax(self.test_y, 1) + 1
        print('---------Accuracy_Test_ELM_y----------:\n',NewIndex_test.shape)
        ind = np.where(NewIndex_test==OriIndex_test)
        self.TestAccuracy = np.floor(len(ind[0]))/len(self.test_y)
               
        
#----print ------------------------------------------------------------
    def Printf(self):
        print('--------train accuracy----------:\n',self.TrainAccuracy)
        print('--------test accuracy-----------:\n',self.TestAccuracy)
        print('--------run time----------------:\n ',self.runtime)
        
#---------evaluate
#---------------------evaluate-------------设 negative = 0  positive = 1----------------------------------
    def Evaluate_Test(self,test_y):

        test_py = []   #存放预测的标签
        
        for i in range(len(self.Test_elm_predict)):         
            if self.Test_elm_predict.argmax(axis=1)[i] == 1:
                test_py.append(1)
            else:
                test_py.append(0)
        
        
        test_y = test_y.astype('int')
        print('self.test_y',test_y.T)
        print('self.test_py',test_py)
        TP, FN, FP, TN, beta = 0, 0, 0, 0, 1
        
        for k in range(0, len(test_y)):
            if test_y[k] == 1 and test_py[k] == 1:
                TP += 1
            if test_y[k] == 1 and test_py[k] == 0:
                FN += 1
            if test_y[k] == 0 and test_py[k] == 1:
                FP += 1
            if test_y[k] == 0 and test_py[k] == 0:
                TN += 1
       
        if TP + FP == 0:
            Accuracy = float(TP + TN) / (TP + FP + FN + TN)
            Precision = 1
            Recall = 0
            F_Measure = 0
            G_mean = 0
      
        elif float(TP) / (TP + FP) + float(TP) / (TP + FN) == 0:
            Accuracy = float(TP + TN) / (TP + FP + FN + TN)
            Precision = 0
            Recall = 0
            F_Measure = 0
            G_mean = 0
       
        else:
            Accuracy = float(TP + TN) / (TP + FP + FN + TN)
            Precision = float(TP) / (TP + FP)
            Recall = float(TP) / (TP + FN)
            F_Measure = float((1 + beta * beta) * Recall * Precision) / (beta * beta * Recall + Precision)
            G_mean = np.sqrt(float(TP) / (TP + FN) * float(TN) / (TN + FP))
        print('TP:\n',TP)
        print('FP:\n',FP)
        print('TN:\n',TN)
        print('FN:\n',FN)
        print('Accuracy:\n',Accuracy)
        print('Precision:\n',Precision)
        print(' Recall:\n', Recall)
        print('F_Measure:\n',F_Measure)
        print('G_mean',G_mean)
        auc = metrics.roc_auc_score(test_py, test_y)
        print('auc:',auc)
#------------------------------------------------------------------------------
###############################################################################
##########################get data###########################################    
############################################################################### 
def read_data(filename):
    f = open(filename)
    line = f.readline()

    data_list = []
    while line:
        line = line.strip()
        num = list(map(str,line.split(' ')))
        num = list(map(str,line.split(',')))
        data_list.append(num)
        line = f.readline()
    f.close()
    
    data_list.pop(0)
    
    print('data_list column num:',len(data_list[0]))
    
    return data_list       #视情况data_list.pop(0)
      
def transfer_labelname(data_list):
    n = 0
    p = 0
    for i in range(len(data_list)):
        
        if data_list[i][-1] == 'negative':             #多数类
            n += 1
            data_list[i][-1] = 0
        else:
            p += 1
            data_list[i][-1] = 1
    
    print('majority:',n)
    print('minority:',p)
    
    IR = n/p
    print('Imbalance ratio:',IR )
            
    return data_list


def get_pure_data(data_list):
    
    frame_data = pd.DataFrame(data_list,columns=['f_1','f_2','f_3','f_4','f_5','f_6','label'])    #columns是列名
    
    #frame_data = pd.DataFrame(data_list)
    pure_data = frame_data.ix[:,0:-1]   #选取标签列前面的列数据,[0:-1]是一个左闭右开的
    #pure_data = np.array(pure_data)
    #pure_label = frame_data.ix[:,[-1]]
    
    return pure_data
    

def get_pure_label(data_list):
    frame_data = pd.DataFrame(data_list,columns=['f_1','f_2','f_3','f_4','f_5','f_6','label'])    #columns是列名
    
    #frame_data = pd.DataFrame(data_list)
    #pure_data = frame_data.ix[:,0:-1]   #选取标签列前面的列数据,[0:-1]是一个左闭右开的
    #pure_data = np.array(pure_data)
    pure_label = frame_data.ix[:,[-1]]
    
    return pure_label

#----------------------需改，暂不用----------------
def get_tran_S(data_list):
   
    
    positive = []
    negative = []
    for i in range(len(data_list)):
        if  data_list[i][-1] == 0:
            negative.append(data_list[i])
        else:
            positive.append(data_list[i])
        
    
    ne = []
    po = []
    for i in range(len(negative)):
        if i< 88:
            ne.append(negative[i])
        
    for j in range(len(positive)):
        if j < 8:
            po.append(positive[j])
            
    ne = np.array(ne)
    po = np.array(po)
    tran_s = np.concatenate((ne,po),axis = 0) 

    return tran_s
###############################################################################    
############################################################################### 



        

###############################################################################    
###############################################################################  
###############################################################################  
#filename2 = r'C:\Users\Administrator\Desktop\data\ecoli-0-1-4-6_vs_5\ecoli-0-1-4-6_vs_5.txt'
#filename = r'C:\Users\Administrator\Desktop\data\ecoli-0-1_vs_5\ecoli-0-1_vs_5.txt'
#
#aa = transfer_labelname(read_data(filename))
#tranA = np.array(get_pure_data(aa)).astype('float')
#labelA  = np.array(get_pure_label(aa)).astype('float')
#
#b = transfer_labelname(read_data(filename2))
#S = get_tran_S(b)
#tranS = np.array(get_pure_data(S)).astype('float')
#print('test_x_shape:\n',tranS.shape)
#labelS  = np.array(get_pure_label(S))
#
##print('labelS:\n',labelS)
#
#test = transfer_labelname(read_data(filename2))
#testS = np.array(get_pure_data(test)).astype('float')
#print('train_x_shape:\n',testS.shape)
#testSlabel = np.array(get_pure_label(test))
#
#
#
#    
#a = Traditional_ELM(train_x=testS,test_x=tranS,train_y=testSlabel,test_y=labelS,NumHiddenNotes=60)
#a.Parameter()
#a.Activation('sigmoid',testS)
#a.weight_(testSlabel,n=260,p=20)
#a.Train_ELM('lp')
#a.Accuracy_Train_ELM('sigmoid')
#a.Test_ELM()
#a.Accuracy_Test_ELM('sigmoid')
#a.Printf()
#a.Evaluate_Test(labelS)
    


################################################################################    
######################dataset2###################################  
################################################################################  
#path1 = r'F:\学术\ZP\00迁移极限学习机\程序TL-ELM\TWELM\difimbatch1.mat'
#path2 = r'F:\学术\ZP\00迁移极限学习机\程序TL-ELM\TWELM\difimbatch2.mat'
#
#def get_batch_data_label(path):
#    
#    file = scio.loadmat(path)
#    x = file['difimbal']   
#    frame_x = pd.DataFrame(x)
#    columns_num = frame_x.columns.size
#    pure_data = np.array(frame_x.iloc[:,1:columns_num-1])
#    pure_label = np.array(frame_x.ix[:,[0]])
#    
#   
#    
#    #转换标签名称，统计非平衡率
#    n = 0
#    p = 0
#   
#    for i in range(len(pure_label)):
#        if pure_label[i] == 1:
#            n += 1
#            pure_label[i] = 0
#        else:
#            p += 1
#            pure_label[i] = 1
#            
#    print('IR:',n/p)
#    print('positive num:',p)
#    print('negative num',n)
#    
#    return pure_data,pure_label
#    
#
#def select_S_data(path):
#    
#    file = scio.loadmat(path)
#    x = file['difimbal']   
#    frame_x = pd.DataFrame(x)
#    
#    po = []
#    ne = []
#
#    pp = []
#    nn = []
#
#    array_x = np.array(frame_x)
#
#    for i in range(len(array_x)):
#        if array_x[i][0] == 1:
#            array_x[i][0] = 0
#            ne.append(array_x[i])
#        else:
#            array_x[i][0] = 1
#            po.append(array_x[i])
#        
#        
#    for j in range(len(ne)):
#        if j < 26:                  #j值根据抽取情况自定义
#            nn.append(ne[j])
#            
#    for k in range(len(po)):
#        if k < 18:                  #k值根据抽取情况自定义
#            pp.append(po[k])
#    
#    frame_nn = pd.DataFrame(nn)
#    columns_nn_num = frame_nn.columns.size
#    pure_nn_data = np.array(frame_nn.iloc[:,1:columns_nn_num-1])
#    pure_nn_label = np.array(frame_nn.ix[:,[0]])
#    
#    frame_pp = pd.DataFrame(pp)
#    columns_pp_num = frame_pp.columns.size
#    pure_pp_data = np.array(frame_pp.iloc[:,1:columns_pp_num-1])
#    pure_pp_label = np.array(frame_pp.ix[:,[0]])
#    
#    tranS_data = np.concatenate((pure_nn_data, pure_pp_data), axis=0)
#    tranS_label = np.concatenate((pure_nn_label, pure_pp_label), axis=0)
#            
#    return tranS_data,tranS_label
#
#
#tranA,labelA = get_batch_data_label(path1)
#testS,testSlabel = get_batch_data_label(path2)
#tranS,labelS = select_S_data(path2)
#
#a = Traditional_ELM(train_x=testS,test_x=tranS,train_y=testSlabel,test_y=labelS,NumHiddenNotes=60)
#a.Parameter()
#a.Activation('sigmoid',testS)
#a.weight_(testSlabel,n=264,p=181)
#a.Train_ELM('lp')
#a.Accuracy_Train_ELM('sigmoid')
#a.Test_ELM()
#a.Accuracy_Test_ELM('sigmoid')
#a.Printf()
#a.Evaluate_Test(labelS)
#    
