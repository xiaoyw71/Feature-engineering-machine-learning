# -*- coding: utf-8 -*-
'''
Created on 2020年12月29日

@author: xiaoyw
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from DataBase.Car_Info import Gas_Collection

class ComboData(object):
    def __init__(self,datas):
        self.df = datas
        # 统一量纲为小时
        self.df['fuel_time'] = round(self.df['fuel_time']/60,2) 
        # 进出站间隔时间    
        self.df['Plate_interva'] = self.df['Entry_interva'] + self.df['Dep_interva'] + self.df['fuel_time']
        self.df['Entry_pre'] = round(self.df['Entry_interva']/self.df['Plate_interva'],3)*100 
        self.df['Dep_pre'] = round(self.df['Dep_interva']/self.df['Plate_interva'],3)*100         


    def  set_datas(self,times=1):
        self.df.sort_values(by=['IC_ID','fuelle_date'],inplace=True,ascending=[True,True])
        # 类似SQL的having，筛选分组结果
        df_ID = self.df.groupby(['Flag','IC_ID', 'License_plate_R'], as_index=False).filter(lambda x:len(x)>times).groupby(['Flag','IC_ID', 'License_plate_R'], as_index=False).size()
        df_ID = df_ID[['Flag','IC_ID', 'License_plate_R']]
        df_feature = self.df[['Flag','IC_ID','License_plate_R','gas_station','year','month','day','weekday','IC_time','fuel_time','Entry_time','Shop_time','Dep_time']]
        #df_feature = df_feature.rename(columns={'IC_time':'fuel_time','License_plate_R':'Lic_plate'})
        df_feature = df_feature.rename(columns={'License_plate_R':'Lic_plate'})
        
        y_df = df_ID[['Flag']]
        self.y_=pd.DataFrame(columns=('TrueIC','FalseIC'))
        self.y_['TrueIC']=y_df['Flag'].apply(lambda x:0 if x==0 else 1)
        self.y_['FalseIC']=y_df['Flag'].apply(lambda x:1 if x==0 else 0)
        self.y_= self.y_.values
        
        # 取20个行数据，不足补零
        x_data = []
        #cols_name = ['IC_ID','Lic_plate','gas_station','year','month','day','weekday','fuel_time','Entry_time','Shop_time','Dep_time']        
        cols_name = ['IC_ID','Lic_plate','gas_station','year','month','day','weekday','IC_time','fuel_time','Entry_time','Shop_time','Dep_time']
        for index,row in df_ID.iterrows():
            IC_ID = row['IC_ID']
            Lic_plate = row['License_plate_R']
    
            df_tmp = df_feature[(df_feature['IC_ID']==IC_ID) & (df_feature['Lic_plate']==Lic_plate)][cols_name]                
            num = len(df_tmp)
            if num > 20:
                num = num - 1               
                df_tmp = df_tmp.iloc[(num - 20):num]
            else:
                idx = [i for i in range(20-num)]
                df_null = pd.DataFrame(0,columns=cols_name,index=idx)
                df_null[['IC_ID','Lic_plate']] = IC_ID , Lic_plate
                df_tmp = pd.concat([df_tmp,df_null],axis=0,ignore_index=True).reset_index()
                df_tmp = df_tmp.drop('index',axis=1)
            
            a = df_tmp.values
            #print(df_tmp)
            a = a.reshape(a.shape[0]*a.shape[1],)
            x_data.append(a)
        
        x_data= np.array(x_data)
        
        #x_data.reshape(-1,220)
        print(x_data.shape)
        #
        self.X = x_data
        # 归一化处理
        MM = MinMaxScaler()
        self.X = MM.fit_transform(x_data)        
        #print(self.X)
        return 

    def  set_datas2(self,times=1):
        #self.df = self.df.drop(drop_cols_name,axis=1)
        #self.df = self.df[self.df['IC_ID_R']>=0].reset_index()
        self.df.sort_values(by=['IC_ID','fuelle_date'],inplace=True,ascending=[True,True])
        
        #df_ID = self.df.groupby(['Flag','IC_ID', 'License_plate'], as_index=False)
        # 类似SQL的having，筛选分组结果
        df_ID = self.df.groupby(['Flag','IC_ID', 'License_plate_R'], as_index=False).filter(lambda x:len(x)>times).groupby(['Flag','IC_ID', 'License_plate_R'], as_index=False).size()
        df_ID = df_ID[['Flag','IC_ID', 'License_plate_R']]
        df_feature = self.df[['Flag','IC_ID','License_plate_R','gas_station','year','month','day','weekday','IC_time','Entry_time','Shop_time','Dep_time']]
        df_feature = df_feature.rename(columns={'IC_time':'fuel_time','License_plate_R':'Lic_plate'})
        
        y_df = df_ID[['Flag']]
        self.y_= y_df
        
        # 取20个行数据，不足补零
        x_data = []
        cols_name = ['gas_station','year','month','day','weekday','fuel_time','Entry_time','Shop_time','Dep_time']
        for index,row in df_ID.iterrows():
            IC_ID = row['IC_ID']
            Lic_plate = row['License_plate_R']
    
            df_tmp = df_feature[(df_feature['IC_ID']==IC_ID) & (df_feature['Lic_plate']==Lic_plate)][cols_name]                
            num = len(df_tmp)
            if num > 20:
                num = num - 1               
                df_tmp = df_tmp.iloc[(num - 20):num]
            else:
                idx = [i for i in range(20-num)]
                df_null = pd.DataFrame(0,columns=cols_name,index=idx)
                #df_null = df_null.drop(index)
                #df_null.fillna(0)
                df_tmp = pd.concat([df_tmp,df_null],axis=0,ignore_index=True).reset_index()
                df_tmp = df_tmp.drop('index',axis=1)
                #df_tmp = df_tmp.append(df_null,ignore_index=True).reset_index()
            
            #x_data.append(df_tmp.values)
            a = df_tmp.values
            a = a.reshape(a.shape[0]*a.shape[1],)
            x_data.append(a) #a.tolist())    
        
        x_data= np.array(x_data)
        
        #x_data.reshape(-1,220)
        print(x_data.shape)
        #
        self.X = pd.DataFrame(x_data)
        #print(self.X)
        return 
    
    def get_datas(self):
        return self.X, self.y_
                                  
def generate_combo_data(times = 0):
    GC = Gas_Collection('study')
    
    df = GC.get_combo_data()
    
    # 'combos' 进出站时间内，发生IC加油的次数
    #drop_cols_name = ['Entry_date', 'Departure_date']
    print(df.columns)  
    # 探索性数据分析
    EDA = ComboData(df)
    
    #EDA.correlation_analysis(cols_name)
    EDA.set_datas(times)
    
if __name__ == '__main__':
    # 1 生成数据并分析
    #generate_combo_data(times = 0)
    # 2 分析数据
    generate_combo_data(1)
    
    pass    
     