# -*- coding: utf-8 -*-
'''
Created on 2020年12月21日

@author: 肖永威
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#多维特征数据进行聚类分析
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures
from DataBase.Car_Info import Gas_Collection
from matplotlib import cm

class ExploreDataAnalysis(object):
    def __init__(self,datas):
        '''
        Constructor
        ''' 
        self.df = datas
        # 统一量纲为小时
        self.df['fuel_time'] = round(self.df['fuel_time']/60,2) 
        # 进出站间隔时间    
        self.df['Plate_interva'] = self.df['Entry_interva'] + self.df['Dep_interva'] + self.df['fuel_time']
        self.df['Entry_pre'] = round(self.df['Entry_interva']/self.df['Plate_interva'],3)*100 
        self.df['Dep_pre'] = round(self.df['Dep_interva']/self.df['Plate_interva'],3)*100         
           
    def correlation_analysis(self,cols_name=[]):
        #names = ['price','fuelle','amount','Payment','vol','changes',
        #           'fuel_month','fuel_day','changes_month','changes_day','fuel_interva','time_before']
        fig = plt.figure() #调用figure创建一个绘图对象
        ax = fig.add_subplot(111)
        cols_num = len(cols_name)
                
        if cols_num >1:
            df = self.df[cols_name]
        else:
            df = self.df
            # 获取表默认列的数量
            cols_name = df.columns
            cols_num = len(df.columns)
            
        ax.set_xticklabels(cols_name) #生成x轴标签
        ax.set_yticklabels(cols_name)                        
            
        correlations = df.corr(method='pearson',min_periods=1)  #计算变量之间的相关系数矩阵
        correlations.to_excel('dcorr1.xlsx')
        print(correlations)
        # plot correlation matrix

        cax = ax.matshow(correlations,cmap = 'inferno', vmin=-1, vmax=1)  #绘制热力图，从-1到1
        fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条
        ticks = np.arange(0,cols_num,1) #生成0-9，步长为1
        ax.set_xticks(ticks)  #生成刻度
        ax.set_yticks(ticks)

        plt.show()
        
        return correlations
    
    def  Features_extra(self,drop_cols_name=[],times=1):
        df = self.df.drop(drop_cols_name,axis=1)

        # 取在同日、同一加油站相遇的次数
        # groupby中的as_index=False，对于聚合输出，返回以组标签作为索引的对象。仅与DataFrame输入相关。as_index = False实际上是“SQL风格”的分组输出。
        df_feature = df.groupby(['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag'], as_index=False)['fuel_interva'].count()        
        #print(df_feature1)
        #df_feature = df_feature1[['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag','fuel_time']]
        df_feature = df_feature.rename(columns={'fuel_interva':'times'})
        # 取相遇次数>times的数据集
        df_feature = df_feature[df_feature['times']>times].reset_index(drop=True) 
        # 依据相遇次数，筛选数据集
        df = pd.merge(df,df_feature[['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag']],how='inner',on=['IC_ID', 'License_plate', 
                            'IC_ID_R', 'License_plate_R', 'Flag'],right_index=True)
        print(df)
        print(df_feature)
        # 取均值
        df_feature1 = df[['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag','fuel_time','IC_time','Shop_time','Entry_time','Dep_time', 'Entry_interva','Shop_interva', 
                          'Dep_interva','combos','Plate_interva','Entry_pre','Dep_pre']].groupby(['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag'], as_index=False).mean()
        df_feature = pd.merge(left=df_feature, right=df_feature1,how="left",on=['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag'])   #左连接 
        print(df_feature.dtypes)
        # 取标准差
        df_feature1 = df[['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag','fuel_time','IC_time','Shop_time','Entry_time','Dep_time', 'Entry_interva', 'Shop_interva',
                          'Dep_interva','Plate_interva','Entry_pre','Dep_pre']].groupby(['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag'], as_index=False).std()
                                  
        print(df_feature1.dtypes)
        #修改为标准差列名
        df_feature1 = df_feature1.rename(columns={'fuel_time':'fuel_time_std','IC_time':'IC_time_std','Entry_time':'Entry_time_std','Dep_time':'Dep_time_std', 
                            'Entry_interva':'Entry_interva_std', 'Dep_interva':'Dep_interva_std', 'Plate_interva':'Plate_interva_std','Shop_time':'Shop_time_std',
                            'Entry_pre':'Entry_pre_std','Dep_pre':'Dep_pre_std','Shop_interva':'Shop_interva_std'})
        print(df_feature1.dtypes)
        df_feature = pd.merge(left=df_feature, right=df_feature1,how="left",on=['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag'])   #左连接                  
        print(df_feature1.dtypes)
        '''
        # 取最大值
        df_feature1 = df[['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag','Shop_interva','Entry_interva', 
                          'Dep_interva']].groupby(['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag'], as_index=False).max()
                                  
        #修改为最大值列名
        df_feature1 = df_feature1.rename(columns={'Entry_interva':'Entry_interva_max', 'Dep_interva':'Dep_interva_max','Shop_interva':'Shop_interva_max'})
        df_feature = pd.merge(left=df_feature, right=df_feature1,how="left",on=['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag'])   #左连接           
        # 取最小值
        df_feature1 = df[['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag','Entry_interva', 
                          'Dep_interva']].groupby(['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag'], as_index=False).min()
                                  
        #修改为最小值列名
        df_feature1 = df_feature1.rename(columns={'Entry_interva':'Entry_interva_min', 'Dep_interva':'Dep_interva_min'})
        df_feature = pd.merge(left=df_feature, right=df_feature1,how="left",on=['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag'])   #左连接           
             
        # 偏度（三阶）
        if times>=5:
            df_feature3 = df[['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag','fuel_time','IC_time','Entry_time','Dep_time', 'Entry_interva', 
                          'Dep_interva','combos','Plate_interva','Entry_pre','Dep_pre' ]].groupby(['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag'], as_index=False).skew()
            df_feature3 = df_feature3.rename(columns={'fuel_time':'fuel_time_skew','IC_time':'IC_time_skew','Entry_time':'Entry_time_skew',
                            'Dep_time':'Dep_time_skew', 'Entry_interva':'Entry_interva_skew', 'Dep_interva':'Dep_interva_skew','combos':'combos_skew',
                            'Plate_interva':'Plate_interva_skew','Entry_pre':'Entry_pre_skew','Dep_pre':'Dep_pre_skew'})
            df_feature = pd.merge(left=df_feature, right=df_feature3,how="left",on=['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag'])   #左连接        
    
            for index,row in df_feature.iterrows():
                IC_ID = row['IC_ID']
                License_plate = row['License_plate']
                IC_ID_R = row['IC_ID_R']
                License_plate_R = row['License_plate_R']
    
                df_tmp = df[(df['IC_ID']==IC_ID) & (df['License_plate']==License_plate) & (df['IC_ID_R']==IC_ID_R) & (df['License_plate_R']==License_plate_R)][['fuel_time','IC_time',
                                'Entry_time','Dep_time', 'Entry_interva', 'Dep_interva','combos','Plate_interva','Entry_pre','Dep_pre']]
    
    
                k4 = df_tmp.kurt()
                df_feature.loc[index:index,('fuel_time_kurt','IC_time_kurt', 'Entry_time_kurt','Dep_time_kurt', 'Entry_interva_kurt', 'Dep_interva_kurt', 'combos_kurt',
                                'Plate_interva_kurt','Entry_pre_kurt','Dep_pre_kurt')] = k4.tolist()        
        '''
        print(df_feature)
        self.data_feature = df_feature

        df_feature = df_feature.drop(['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R'],axis=1)
               
        print(df_feature.columns)
        #df_feature = df_feature[df_feature['times']>1].reset_index(drop=True) 
        print(df_feature)
        self.df_feature = df_feature
        
        return df_feature
    
    def get_Data_Feature(self):
        return self.data_feature
    # 获取结果集的列名
    def get_Feature_Columns(self):
        
        return self.df_feature.columns
 
    def set_DataAnalysis(self,datas):
        
        self.data_feature = datas
        self.df_feature = datas.drop(['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R'],axis=1)
        
        return
        
    def show_heatmap(self):
        fig = plt.figure() #调用figure创建一个绘图对象
        ax = fig.add_subplot(111)
        cols_name = self.df_feature.columns
        cols_num = len(self.df_feature.columns)
        correlations = self.df_feature.corr(method='pearson',min_periods=1)  #计算变量之间的相关系数矩阵
        self.corr = correlations
        correlations.to_excel('dcorr1.xlsx')
        print(correlations)
        # plot correlation matrix

        cax = ax.matshow(correlations,cmap = 'inferno', vmin=-1, vmax=1)  #绘制热力图，从-1到1
        fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条
        ticks = np.arange(0,cols_num,1) #生成0-9，步长为1
        ax.set_xticks(ticks)  #生成刻度
        ax.set_yticks(ticks)
        ax.set_xticklabels(cols_name) #生成x轴标签
        ax.set_yticklabels(cols_name)                
        plt.show()
                
        return correlations
    
    def important_feature(self,cols_name):  
        df = self.df_feature[cols_name].fillna(0)
        print(df.dtypes)
        print(df[df.isnull().T.any()])
        
        y_df = df[['Flag']]
        X_df = df.drop('Flag',axis=1)
        #注意训练集、测试集返回参数顺序
        x_train,x_text, y_train, y_test = train_test_split(X_df,y_df,test_size=0.1)
    
        #y_test = df_test_y.values
        # n_estimators：森林中树的数量,随机森林中树的数量默认10个树，精度递增显著，但并不是越多越好，加上verbose=True，显示进程使用信息
        # n_jobs  整数 可选（默认=1） 适合和预测并行运行的作业数，如果为-1，则将作业数设置为核心数
        forest_model = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=-1)
        forest_model.fit(x_train, y_train)
        feat_labels = X_df.columns
        #feat_labels = col_names[1:]
        # 下面对训练好的随机森林，完成重要性评估
        # feature_importances_  可以调取关于特征重要程度
        importances = forest_model.feature_importances_
        print("重要性：", importances)
        x_columns = X_df.columns
        #x_columns = col_names[1:]
        indices = np.argsort(importances)[::-1]
        x_columns_indices = []
        for f in range(x_train.shape[1]):
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
        plt.title("IC卡加油与号牌识别集中各个特征的重要程度", fontsize=16)
        plt.ylabel("import level", fontsize=12, rotation=90)
    
        num = len(x_columns)
        for i in range(num):
            plt.bar(i, importances[indices[i]], color='blue', align='center')
            plt.xticks(np.arange(num), x_columns_indices, rotation=90, fontsize=12)
        plt.tight_layout()
        
        plt.show()   
        
        return importances
    
    def cluster_analysis(self):
        df = self.df_feature.fillna(0)
        
        Flag=list(df['Flag'])
        #print(grain_variety)
        df=df.drop('Flag',axis=1)
        print(df.columns)
        print(df)
        
        # 升纬没有效果
        #poly = PolynomialFeatures(degree=2,include_bias = False)
        #X = df.values
        #samples = poly.fit_transform(X)
        # df = df[['times']]
        samples=df.values
        #samples=df_poly.values
        print(samples)
        
        #标准化
        scaler=StandardScaler()
        
        kmeans=KMeans(n_clusters=2,random_state=9,precompute_distances='auto',max_iter=1000)
        pipeline=make_pipeline(scaler,kmeans)
        pipeline.fit(samples) #训练模型
        labels=pipeline.predict(samples)#预测
        
        df_cluster=pd.DataFrame({'labels':labels,'Flag':Flag})
        ct=pd.crosstab(df_cluster['labels'],df_cluster['Flag'])
        print('K-Means')
        print(ct)

        #标准化
        scaler=StandardScaler()        
        Minikmeans=MiniBatchKMeans(n_clusters=2,random_state=9,max_iter=1000)
        print('MiniBatchKMeans')
        pipeline=make_pipeline(scaler,Minikmeans)
        pipeline.fit(samples) #训练模型
        labels=pipeline.predict(samples)#预测        

        df_cluster=pd.DataFrame({'labels':labels,'Flag':Flag})
        ct=pd.crosstab(df_cluster['labels'],df_cluster['Flag'])
        print(ct)
        
        print('Birch')        
        birch = Birch(n_clusters = 2,threshold = 0.6)
        est = birch.fit(samples)
        labels = est.labels_
        df_cluster=pd.DataFrame({'labels':labels,'Flag':Flag})
        ct=pd.crosstab(df_cluster['labels'],df_cluster['Flag'])
        print(ct)
        
        print('GaussianMixture') 
        gmm = GaussianMixture(n_components=2)
        gmm.fit(samples)
        labels = gmm.predict(samples)     
        
        df_cluster=pd.DataFrame({'labels':labels,'Flag':Flag})
        ct=pd.crosstab(df_cluster['labels'],df_cluster['Flag'])
        print(ct)

        return
    
    def dimension_upgrading(self,df,f1):
        
        return df
    
    def draw_corr_bar(self,key_name):        
        values = self.corr.loc[key_name].values.tolist()
        print(values)
        draw_bar(self.df_feature.columns,values)
        
        return
    
    def draw_Hist_KDE(self):
        name = self.df_feature.columns
        k= 0
        fig1, ax1 = plt.subplots(nrows=6, ncols=4)
        for i in range(6):               
            for j in range(4):           
                ax1[i, j].hist(self.df_feature[name[k]],density=True)

                ax1[i, j].set_ylabel(name[k])
                k = k + 1
                            
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
    ax = plt.bar(key_name,key_values,width=0.5,color=colors,edgecolor='black') # edgecolor边框颜色
    
    sm = cm.ScalarMappable(cmap=map_vir,norm=norm)  # norm设置最大最小值
    sm.set_array([])
    plt.colorbar(sm)
    autolable(ax)
    
    plt.show()

def generate_combo_data(times = 0):
    GC = Gas_Collection('study')
    
    df = GC.get_combo_data()
    
    # 'combos' 进出站时间内，发生IC加油的次数
    drop_cols_name = ['fuelle_date', 'Entry_date', 'Departure_date']
    print(df.columns) 
    print(df.dtypes) 
    # 探索性数据分析
    EDA = ExploreDataAnalysis(df)
    
    #EDA.correlation_analysis(cols_name)
    df_feature = EDA.Features_extra(drop_cols_name,times)
    print(df_feature.dtypes)
    
    GC.generate_Analysis_data('Analysis_data1', EDA.get_Data_Feature())
      
    # 展现热力图
    EDA.show_heatmap()
    
    EDA.draw_corr_bar('Flag')
  
    cols_name = ['Flag', 'times','fuel_time','IC_time','Entry_time','Dep_time', 'Entry_interva', 'Dep_interva','combos',
                            'fuel_time_std','IC_time_std','Entry_time_std','Dep_time_std', 'Entry_interva_std', 'Dep_interva_std','combos_std',
                            'fuel_time_kurt','IC_time_kurt', 'Entry_time_kurt','Dep_time_kurt', 'Entry_interva_kurt', 'Dep_interva_kurt','combos_kurt']       

    cols_name = ['Flag', 'times','fuel_time','IC_time','Entry_time','Dep_time', 'Entry_interva', 'Dep_interva','combos',
                            'fuel_time_std','IC_time_std','Entry_time_std','Dep_time_std', 'Entry_interva_std', 'Dep_interva_std',
                            'Plate_interva','Entry_pre','Dep_pre','Plate_interva_std','Entry_pre_std','Dep_pre_std','Shop_time',
                            'Shop_interva','Shop_interva_std']
                            #'Entry_interva_max', 'Dep_interva_max','Entry_interva_min', 'Dep_interva_min'] #,'Shop_interva'] #,'Shop_interva_std','Shop_interva_max']   

    cols_name = EDA.get_Feature_Columns()

    EDA.important_feature(cols_name)
    EDA.cluster_analysis()  
    
    return

def ExploreDataAnalysis_combo_data(dbname = 'Analysis_data'):      
    GC = Gas_Collection('study')
    
    df = GC.get_combo_data()
    
    # 'combos' 进出站时间内，发生IC加油的次数
    drop_cols_name = ['fuelle_date', 'Entry_date', 'Departure_date']
    print(df.columns)  
    # 探索性数据分析
    EDA = ExploreDataAnalysis(df)
    
    cols_name = ['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R',
                    'Flag', 'times', 'fuel_time', 'IC_time', 'Shop_time', 'Entry_time',
                    'Dep_time', 'Entry_interva', 'Shop_interva', 'Dep_interva', 'combos',
                    'Plate_interva', 'Entry_pre', 'Dep_pre', 'fuel_time_std', 'IC_time_std',
                    'Shop_time_std', 'Entry_time_std', 'Dep_time_std', 'Entry_interva_std',
                    'Shop_interva_std', 'Dep_interva_std', 'Plate_interva_std',
                    'Entry_pre_std', 'Dep_pre_std']
    datas = GC.get_Analysis_data(dbname, cols_name)
    EDA.set_DataAnalysis(datas)
    
    # 展现热力图
    EDA.show_heatmap()
    
    EDA.draw_corr_bar('Flag')
  
    cols_name = ['Flag', 'times','fuel_time','IC_time','Entry_time','Dep_time', 'Entry_interva', 'Dep_interva','combos',
                            'fuel_time_std','IC_time_std','Entry_time_std','Dep_time_std', 'Entry_interva_std', 'Dep_interva_std','combos_std',
                            'fuel_time_kurt','IC_time_kurt', 'Entry_time_kurt','Dep_time_kurt', 'Entry_interva_kurt', 'Dep_interva_kurt','combos_kurt']       

    cols_name = ['Flag', 'times','fuel_time','IC_time','Entry_time','Dep_time', 'Entry_interva', 'Dep_interva','combos',
                            'fuel_time_std','IC_time_std','Entry_time_std','Dep_time_std', 'Entry_interva_std', 'Dep_interva_std',
                            'Plate_interva','Entry_pre','Dep_pre','Plate_interva_std','Entry_pre_std','Dep_pre_std','Shop_time',
                            'Shop_interva','Shop_interva_std']
                            #'Entry_interva_max', 'Dep_interva_max','Entry_interva_min', 'Dep_interva_min'] #,'Shop_interva'] #,'Shop_interva_std','Shop_interva_max']   

    cols_name = EDA.get_Feature_Columns()

    #EDA.important_feature(cols_name)
    EDA.draw_Hist_KDE()
    EDA.cluster_analysis()
    
    return

    
if __name__ == '__main__':
    # 1 生成数据并分析
    #generate_combo_data(times = 0)
    # 2 分析数据
    ExploreDataAnalysis_combo_data('Analysis_data1')
    
    pass
