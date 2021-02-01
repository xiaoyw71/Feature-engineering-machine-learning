# -*- coding: utf-8 -*-
'''
Created on 2021年1月26日

@author: xiaoyw
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import plot_importance
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from matplotlib import cm
import seaborn as sns

#导入此解决方案中重要的模块SMOTE用来生成oversample样本
from imblearn.over_sampling import SMOTE

#设置查看列不省略
pd.set_option( 'display.max_columns', None)

class CustomerData(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
    def read_data(self):
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        #print(df.dtypes)
        #查看数据是否存在Null，没有发现。
        nulldata = pd.isnull(df).sum()
        print(nulldata)
        # 查看数据类型，根据一般经验，发现‘TotalCharges’总消费额的数据类型为字符串，应该转换为浮点型数据。
        #'TotalCharges'存在缺失值，强制转换为数字，不可转换的变为NaN
        df['TotalCharges']=pd.to_numeric(df['TotalCharges'], errors='coerce') #astype('float64')
        self.df = df
        print(df.dtypes)
        colsname = df.columns.to_list()
        #客户ID没有分析意义
        colsname.remove('customerID')
        self.CData = df[colsname].copy() 
                
        #print(colsname)
        #'TotalCharges'存在缺失值，强制转换为数字，不可转换的变为NaN
        #self.CData['TotalCharges']=pd.to_numeric(self.CData['TotalCharges'], errors='coerce') #astype('float64')
        #将总消费额填充为月消费额
        self.CData['TotalCharges'] = self.CData[['TotalCharges']].apply(lambda x: x.fillna(self.CData['MonthlyCharges']),axis=0)
        #查看是否替换成功
        #print(self.CData[self.CData['tenure']==0][['tenure','MonthlyCharges','TotalCharges']])
        # 将‘tenure’入网时长从0修改为1
        self.CData['tenure'] = self.CData['tenure'].apply(lambda x:1 if x==0 else x)
        #print(self.CData[self.CData['tenure']==1])
        self.CData['Contract']=self.CData['Contract'].map({'Month-to-month':1,'One year':12,'Two year':24})
        self.CData['Contract']=pd.to_numeric(self.CData['Contract'], errors='coerce')
        print(self.CData.dtypes)

        Cols = [c for c in self.CData.columns if self.CData[c].dtype == 'object' or c == 'SeniorCitizen']
        Cols.remove('Churn')      
        #print(self.CData.dtypes)
        # 对于离散特征，特征之间没有大小关系，采用one-hot编码；特征之间有大小关联，则采用数值映射。
        for col in Cols:
            if self.CData[col].nunique() == 2:
                self.CData[col] = pd.factorize(self.CData[col])[0]
            else:
                self.CData = pd.get_dummies(self.CData, columns=[col])        
        
        #self.CData['gender']=self.CData['gender'].map({'Male':1,'Female':0})
        self.CData['Churn']=self.CData['Churn'].map({'Yes':1,'No':0})
        self.CData['AverageChargs'] = self.CData['TotalCharges']/self.CData['tenure']

        #print(self.CData)
        #print(self.CData.dtypes)
    # 联系变量离散化
    def discrete_data(self):
        print('qcut')
        self.CData['tenure']=pd.qcut(self.CData['tenure'],6,labels=[1,2,3,4,5,6])
        #self.CData['tenure'] = self.CData['tenure'].map({1:1,2:2,3:3,4:4,5:5,6:6})

        k =self.CData['MonthlyCharges'].describe() 
        #k = self.CData['tenure'].describe() 

        print(k)
        #用四分位数进行离散
        self.CData['MonthlyCharges']=pd.qcut(self.CData['MonthlyCharges'],4,labels=[1,2,3,4])
        
        k = self.CData['TotalCharges'].describe()
        print('discrete data')
        print(self.CData.dtypes)        
        self.CData['TotalCharges']=pd.qcut(self.CData['TotalCharges'],4,labels=[1,2,3,4])
        #self.CData['TotalCharges'].describe()
        print(self.CData)
    
    
    
    def set_train_data(self):
        X = self.CData.drop(['Churn'],axis=1)
        Y = self.CData[['Churn']]
        self.x_train,self.x_test, self.y_train, self.y_test = train_test_split(X,Y,test_size=0.3)
        
        def set_oversampler(features,labels):
            #利用SMOTE创造新的数据集 
            #初始化SMOTE 模型
            oversampler=SMOTE(random_state=0)
            #使用SMOTE模型，创造新的数据集
            os_features,os_labels=oversampler.fit_sample(features,labels)
            #切分新生成的数据集
            os_features_train, os_features_test, os_labels_train, os_labels_test = train_test_split(os_features, os_labels, test_size=0.01) 
            #return os_features_train, os_features_test, os_labels_train, os_labels_test
            return os_features_train, os_labels_train
        
        #self.x_train, self.y_train = set_oversampler(self.x_train,self.y_train)
        '''
        #看看新构造的oversample数据集中0,1分布情况
        #常用pandas的value_counts确认数据出现的频率
        os_count_classes = pd.value_counts(os_labels['Churn'], sort = True).sort_index()
        os_count_classes.plot(kind = 'bar')
        plt.title("Fraud class histogram")
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        plt.show()
        '''
        
    def forecast_analysis_by_XGB(self):
        params ={'learning_rate': 0.02,
          'max_depth': 8,                # 构建树的深度，越大越容易过拟合
          'num_boost_round':10,
          #'objective': 'multi:softprob', # 多分类的问题
          'objective': 'binary:logistic', # 二分类：binary:logistic
          #'objective': 'binary:logitraw',  # 二分类
          'random_state': 20,
          'silent':0,
          'subsample':0.6,
          'min_child_weight':5,
          #'num_class':2,                 # 类别数，与 multisoftmax 并用
          #'eval_metric':['mlogloss','merror','auc'],   # 多分类情况
          'eval_metric':['logloss','error','auc'], # 二分类情况
          'eta':0.3                      #为了防止过拟合，更新过程中用到的收缩步长。eta通过缩减特征 的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
        }
      
        dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
        dtest = xgb.DMatrix(self.x_test,label=self.y_test)
        
        res = xgb.cv(params,dtrain) #,num_boost_round=5000,metrics='auc',early_stopping_rounds=25)
        #找到最佳迭代轮数
        best_nround = res.shape[0] - 1
        print('best_nround:{}'.format(best_nround))
        if best_nround<10:
            best_nround = 100
        
        watchlist = [(dtrain,'train'),(dtest,'eval')]
        evals_result = {}
        
        model = xgb.train(params,dtrain,num_boost_round=best_nround,evals = watchlist,evals_result=evals_result)
        y_pred=model.predict(xgb.DMatrix(self.x_test))        
        
        #model.save_model('XGboostClass.model')
        #print(y_pred.shape)

        #yprob = np.argmax(y_pred, axis=1)  # return the index of the biggest pro

        #predictions = [round(value) for value in yprob]
        predictions = [round(value) for value in y_pred]
        # 计算准确率
        accuracy = accuracy_score(self.y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        print(params)
        
        print('绘制训练AUC下降趋势图')
        
        #验证数据评估指标，与param参数，'eval_metric':['logloss','error','auc']相关
        #验证包括训练和验证两个部分（train、eval），如上所示3个参数，则是6组数据
        plt.figure(111)
        plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
        plt.grid()      

        plt.plot(evals_result['train']['logloss'],label = 'train-logloss',color='green')
        plt.plot(evals_result['train']['error'],label = 'train-error',color='blue')
        plt.plot(evals_result['train']['auc'],label = 'train-auc',color='coral')
        plt.plot(evals_result['eval']['logloss'],label = 'eval-logloss',color='deeppink')
        plt.plot(evals_result['eval']['error'],label = 'eval-error',color='red')
        plt.plot(evals_result['eval']['auc'],label = 'eval-auc',color='gray')        
        plt.xlabel('训练次数')
        # 显示图例
        plt.legend()
        # 显示重要特征
        plot_importance(model)
        plt.subplots_adjust(left=0.4)
        plt.show()

    def important_feature(self):     
        # n_estimators：森林中树的数量,随机森林中树的数量默认10个树，精度递增显著，但并不是越多越好，加上verbose=True，显示进程使用信息
        # n_jobs  整数 可选（默认=1） 适合和预测并行运行的作业数，如果为-1，则将作业数设置为核心数
        forest_model = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=-1)
        forest_model.fit(self.x_train, self.y_train)
        feat_labels = self.x_train.columns
        #feat_labels = col_names[1:]
        # 下面对训练好的随机森林，完成重要性评估
        # feature_importances_  可以调取关于特征重要程度
        importances = forest_model.feature_importances_
        print("重要性：", importances)
        x_columns = self.x_train.columns
        #x_columns = col_names[1:]
        indices = np.argsort(importances)[::-1]
        x_columns_indices = []
        for f in range(self.x_train.shape[1]):
            # 对于最后需要逆序排序，我认为是做了类似决策树回溯的取值，从叶子收敛
            # 到根，根部重要程度高于叶子。
            print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
            x_columns_indices.append(feat_labels[indices[f]])
         
        print(x_columns_indices)
        print(len(x_columns))
        print(x_columns)
        #print(np.arange(x_columns.shape[0]))
         
        # 筛选变量（选择重要性比较高的变量）
        #threshold = 0.05
        #x_selected = x_train[:, importances > threshold]
             
        plt.figure(figsize=(6, 6))
        plt.rcParams['font.sans-serif'] = ["SimHei"]
        plt.rcParams['axes.unicode_minus'] = False    
        plt.title("客户流失集中各个特征的重要程度", fontsize=16)
        plt.ylabel("import level", fontsize=12, rotation=90)
    
        num = len(x_columns)
        for i in range(num):
            plt.bar(i, importances[indices[i]], color='blue', align='center')
            plt.xticks(np.arange(num), x_columns_indices, rotation=90, fontsize=12)
        plt.tight_layout()
        
        plt.show()   
        
        return importances
    
    def correlationAnalysis(self):
        #plt.figure(figsize=(16,8))
        #self.CData.corr()['Churn'].sort_values(ascending=False).plot(kind='bar')
        #plt.show()
        #corr = self.CData.corr()['Churn'].sort_values(ascending=False) 
        corr = self.CData.corr()['MonthlyCharges'].sort_values(ascending=False) 
        print(corr)
        #print(corr.values)
        draw_bar(corr.index,corr.values)

    def dataVisualization(self):
        plt.rcParams['figure.figsize']= 12,6 #6,6
        plt.subplot(1,2,1)
        plt.pie(self.df['Churn'].value_counts(),labels=self.df['Churn'].value_counts().index,autopct='%1.2f%%',explode=(0.1,0))
        plt.title('Churn(Yes/No) Ratio')
        plt.subplot(1,2,2)
        dd = self.df[['MonthlyCharges','TotalCharges','Churn']].groupby(['Churn'], as_index=False).sum()
        plt.pie(dd['MonthlyCharges'],labels=dd['Churn'],autopct='%1.2f%%',explode=(0.1,0))
        plt.title('Churn(Yes/No) MonthlyCharges')        
        plt.show()
        # 在网时长直方图
        #self.CData[['tenure']].plot.hist()
        #plt.show()
        # 对比Bar图
        self.df['churn_rate'] = self.df['Churn'].replace("No", 0).replace("Yes", 1)
        items=["PhoneService","InternetService"]
        fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,4))
        for i,item in enumerate(items):
            plt.subplot(1,2,(i+1))
            ax=sns.barplot(x=item,y='churn_rate',data=self.df)
            plt.rcParams.update({'font.size': 11})
            plt.xlabel(str(item))
            plt.ylabel("Churn Rate")
            plt.title("Churn By "+str(item))
            i+=1  
        plt.show() 
        
        items=['MonthlyCharges','TotalCharges']
        
        fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,4))
        dd = self.df[['MonthlyCharges','TotalCharges','churn_rate']].groupby(['churn_rate'], as_index=False).sum()

        print(self.df)
        print(dd)
        for i,item in enumerate(items):
            plt.subplot(1,2,(i+1))
            ax=sns.barplot(x=item,y='churn_rate',data=dd) #,estimator=sum)
            plt.rcParams.update({'font.size': 11})
            plt.xlabel(str(item))
            plt.ylabel("Churn Rate")
            plt.title("Churn By "+str(item))
            i+=1  
        plt.show() 
        
        
def draw_bar(key_name,key_values):
    plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
    plt.rcParams['axes.unicode_minus']=False
    # 标准柱状图的值
    def autolable(rects):
        for rect in rects:
            height = rect.get_height()
            if height>=0:
                plt.text(rect.get_x()+rect.get_width()/2.0 - 0.3,height+0.02,'%.3f'%height)
            else:
                plt.text(rect.get_x()+rect.get_width()/2.0 - 0.3,height-0.06,'%.3f'%height)
                # 如果存在小于0的数值，则画0刻度横向直线
                plt.axhline(y=0,color='black')
    #归一化
    norm = plt.Normalize(-1,1)
    norm_values = norm(key_values)
    map_vir = cm.get_cmap(name='inferno')
    colors = map_vir(norm_values)
    fig = plt.figure() #调用figure创建一个绘图对象
    plt.subplot(111)
    plt.xticks(rotation=90)
    
    ax = plt.bar(key_name,key_values,width=0.5,color=colors,edgecolor='black') # edgecolor边框颜色
    
    sm = cm.ScalarMappable(cmap=map_vir,norm=norm)  # norm设置最大最小值
    sm.set_array([])
    plt.colorbar(sm)
    autolable(ax)

    #plt.margins(0.2)
    plt.subplots_adjust(bottom=0.4)    
    plt.show()

        
        
if __name__ == "__main__":
    CDA = CustomerData('')
    CDA.read_data() 
    #CDA.discrete_data() # category 类型转换出现问题？
    #CDA.dataVisualization()
    CDA.set_train_data()
    #importances = CDA.important_feature()
    #print(importances)
    #CDA.correlationAnalysis()
    CDA.forecast_analysis_by_XGB() 
     