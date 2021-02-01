# -*- coding: utf-8 -*-
'''
Created on 2021年1月30日

@author: xiaoyw
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,MiniBatchKMeans
from CustomerChurn.CustomerDataAnalysis import CustomerData

class ChurnAnalysis(object):

    def __init__(self, params):
        self.df = params
        
    def monthlyChargesAnalysis(self):
        dd = self.df[['MonthlyCharges','TotalCharges','Churn']].groupby(['Churn'], as_index=False).sum()
        print(dd)
        key_name = dd['Churn']
        print(key_name)
        key_values = dd['MonthlyCharges']
        ax = plt.bar(key_name,key_values,width=0.5,color='blue',edgecolor='black')
        autolable(plt,ax)
        plt.show()
        
    def kdeAnalysis(self):
        plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
        plt.rcParams['axes.unicode_minus']=False 
        plt.figure(figsize=(18, 5))
        plt.subplot(1,3,1)
        kdeplot('MonthlyCharges','月度消费',self.df,'Churn')
        plt.subplot(1,3,2)
        kdeplot('TotalCharges','总消费额',self.df,'Churn')
        plt.subplot(1,3,3)
        kdeplot('tenure','在网时长',self.df,'Churn')
        plt.show()
        
    def serviceAnalysis(self):
        plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
        plt.rcParams['axes.unicode_minus']=False     
        sns.set(font='SimHei') #, font_scale=0.8)        # 解决Seaborn中文显示问题
        sns.set_style({'font.sans-serif':['simhei', 'Arial']}) 
        self.df['churn_rate'] = self.df['Churn'].replace("No", 0).replace("Yes", 1)
        items=['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','InternetService', 'StreamingMovies']
        items_name=['安全服务','备份业务','保护业务','技术支持','互联网服务','网络电影']
        def get_order(items_index):
            if items_index == 4:
                return ['DSL','Fiber optic','No']
            else:
                return ['Yes','No','No internet service']
        fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(8,12))
        for i,item in enumerate(items):
            plt.subplot(2,3,(i+1))
            ax=sns.barplot(x=item,y='churn_rate',data=self.df,order=get_order(i))
            plt.rcParams.update({'font.size': 12})
            plt.xlabel(str(items_name[i]))
            plt.title(str(items_name[i])+'流失情况')
            i+=1        
        plt.show()
        
    def clusteringAnalysis(self):
        data = self.df[self.df['Churn']==1].reset_index(drop=True)#重设索引
        samples = data.drop(['Churn'],axis=1)
        samples = data[['MonthlyCharges','TotalCharges','AverageChargs','tenure','Contract']]
  
        #标准化
        scaler=StandardScaler()
        
        #kmeans=KMeans(n_clusters=3,random_state=9,precompute_distances='auto',max_iter=1000)
        kmeans=MiniBatchKMeans(n_clusters=3,random_state=9,max_iter=100)
        pipeline=make_pipeline(scaler,kmeans)
        pipeline.fit(samples) #训练模型
        labels=pipeline.predict(samples)#预测 
        samples['labels'] = labels #合并数据集
        print(samples)       
        
    def show_heatmap(self):
        plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
        plt.rcParams['axes.unicode_minus']=False  
        fig = plt.figure() #调用figure创建一个绘图对象
        ax = fig.add_subplot(111)
        cols_name = ['Churn','TotalCharges','MonthlyCharges','tenure','Contract']
        cols_num = len(cols_name)
        correlations = self.df[cols_name].corr(method='pearson',min_periods=1)  #计算变量之间的相关系数矩阵
        self.corr = correlations
        #correlations.to_excel('dcorr1.xlsx')
        print(correlations)
        # plot correlation matrix

        cax = ax.matshow(correlations,cmap = 'inferno', vmin=-1, vmax=1)  #绘制热力图，从-1到1
        fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条
        ticks = np.arange(0,cols_num,1) #生成0-9，步长为1
        ax.set_xticks(ticks)  #生成刻度
        ax.set_yticks(ticks)
        cols_name = ['流失','总消费','月消费','在网时长','合同']
        ax.set_xticklabels(cols_name) #生成x轴标签
        ax.set_yticklabels(cols_name)                
        plt.show()
                
# 自动标准柱状图的值
def autolable(mplt,rects):
    for rect in rects:
        height = rect.get_height()
        if height>=0:
            mplt.text(rect.get_x()+rect.get_width()/2.0 - 0.3,height+0.02,'%.3f'%height)
        else:
            mplt.text(rect.get_x()+rect.get_width()/2.0 - 0.3,height-0.06,'%.3f'%height)
            # 如果存在小于0的数值，则画0刻度横向直线
            mplt.axhline(y=0,color='black')        
        
# Kernel density estimaton核密度估计
def kdeplot(feature,xlabel,data,tag='Churn'):
    #plt.figure(figsize=(8, 6))
    #plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
    #plt.rcParams['axes.unicode_minus']=False    
    plt.title('KDE for {0}'.format(feature))
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    
    sns.set(font='SimHei') #, font_scale=0.8)        # 解决Seaborn中文显示问题
    sns.set_style({'font.sans-serif':['simhei', 'Arial']}) 
    #ax0 = sns.kdeplot(data[data['Churn'] == 'No'][feature],  label= 'Churn: No', shade='True')
    ax0 = sns.kdeplot(data[data['Churn'] == 'No'][feature],  label= '未流失', shade='True',legend=True)
    ax1 = sns.kdeplot(data[data['Churn'] == 'Yes'][feature], label= '流失',shade='True',legend=True)
    plt.xlabel(xlabel)
    plt.rcParams.update({'font.size': 10})
    plt.legend(fontsize=10)

 
if __name__ == '__main__':
    CDA = CustomerData('')
    CDA.read_data()        
    #CA = ChurnAnalysis(CDA.df)  
    #CA.monthlyChargesAnalysis() 
    #CA.kdeAnalysis()
    #CA.serviceAnalysis()
    CA = ChurnAnalysis(CDA.CData)  
    #CA.show_heatmap()
    CA.clusteringAnalysis()