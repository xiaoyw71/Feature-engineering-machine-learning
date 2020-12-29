# -*- coding: utf-8 -*-
'''
Created on 2020年12月18日

@author: 肖永威
'''
from DataBase.Car_Info import Gas_Collection
import pandas as pd
import datetime

class IC_Plate(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.GC = Gas_Collection('study')
    # 组合数据
    def combo_data(self):
        df_IC = self.GC.get_IC_Sales()
        df_Plate = self.GC.get_Plate_record()
        df_Shopping = self.GC.get_Shopping_data()
        df_Shopping = df_Shopping.drop('amount',axis=1)
        
        # 修改两表相同的列名
        df_Plate = df_Plate.rename(columns={'IC_ID':'IC_ID_R','License_plate':'License_plate_R'})
        # 以IC卡加油为基准，不是IC卡的数据，不予以处理
        df_IC = df_IC[(df_IC.IC_ID<1100) & (df_IC.IC_ID>=0)]
        #df_Plate = df_Plate[(df_Plate.IC_ID_R>=0) & (df_Plate.IC_ID_R<=100)]
        # 提取时间特征                
        def get_date_features(data,col_name,name,update_col=False):
            # data:提取日期型和时间型的特征变量
            data[col_name] = data[col_name].astype('datetime64')
            if update_col==False:
                data[name + '_year']= data[col_name].dt.year
                data[name + '_month'] = data[col_name].dt.month
                data[name + '_day'] = data[col_name].dt.day
                data[name + '_weekday'] = data[col_name].dt.weekday       # 周几，0为周一   
            elif update_col:
                data['year']= data[col_name].dt.year
                data['month'] = data[col_name].dt.month
                data['day'] = data[col_name].dt.day
                data['weekday'] = data[col_name].dt.weekday       # 周几，0为周一                  
            data[name + '_hour'] = data[col_name].dt.hour
            data[name + '_minute'] = data[col_name].dt.minute
            data[name + '_second'] = data[col_name].dt.second    # 秒         
        
            return data
        
        df_IC = get_date_features(df_IC,'fuelle_date','IC',True)
        df_Plate = get_date_features(df_Plate,'Entry_date','Entry',True)
        df_Plate = get_date_features(df_Plate,'Departure_date','Dep',False)
        df_Shopping = get_date_features(df_Shopping,'shopping_date','shop',True)
        # 把Shopping时间并入df_IC中
        df_IC = pd.merge(df_IC,df_Shopping,how='left',on=['IC_ID', 'License_plate','year','month','day','weekday','gas_station'])
        print('合并非油消费记录：{}'.format(len(df_IC)))
        
        df_IC = df_IC.drop('_id_y',axis=1)

        #df_IC[['shopping_date','shop_hour','shop_minute', 'shop_second']] = df_IC[['fuelle_date','IC_hour','IC_minute','IC_second']].where(df_IC.shopping_date.isnull() )
        df_IC1 = df_IC[df_IC['shopping_date'].isnull() ].reset_index(drop=True) 
        df_IC2 = df_IC[df_IC['shopping_date'].notnull() ].reset_index(drop=True) 
        df_IC1[['shopping_date','shop_hour','shop_minute', 'shop_second']] = df_IC1[['fuelle_date','IC_hour','IC_minute','IC_second']]
        
        df_IC = pd.concat([df_IC1,df_IC2]).reset_index()
        print('合并整理非油消费记录：{}'.format(len(df_IC)))        
        # 加油间隔周期（天数时长）
        # 指定多列排序(注意：对IC_ID列升序，再对fuelle_date列升序)，ascending不指定的话，默认是True升序
        df_IC.sort_values(by=['IC_ID','fuelle_date'],inplace=True,ascending=[True,True])
        df_IC['fuel_interva'] = df_IC['fuelle_date'].diff().dt.days
        df_IC['fuel_interva'] = df_IC['fuel_interva'].fillna(0)
        # 跨IC号，重新计时间，出现间隔时间为负的情况
        df_IC['fuel_interva'] = df_IC['fuel_interva'].apply(lambda x: x if x>0 else 0)
        velocity = 37 # 标准流速
        df_IC['fuel_time'] = round(df_IC['fuelle']/velocity,2) # 加油量，影响离开时间
        # 进站间隔周期（天数时长）
        df_Plate.sort_values(by=['License_plate_R','Entry_date'],inplace=True,ascending=[True,True])
        df_Plate['Entry_interva'] = df_Plate['Entry_date'].diff().dt.days
        df_Plate['Entry_interva'] = df_Plate['Entry_interva'].fillna(0)
        df_Plate['Entry_interva'] = df_Plate['Entry_interva'].apply(lambda x: x if x>0 else 0)
        
        #print(df_Plate.dtypes)      
        df_IC = df_IC[['IC_ID','License_plate','year','month','day','weekday','gas_station','fuelle_date','fuel_interva','fuel_time',
                       'IC_hour','IC_minute','shopping_date','shop_hour','shop_minute']]

        df_Plate = df_Plate[['IC_ID_R','License_plate_R','year','month','day','gas_station','Entry_date','Entry_interva','Departure_date',
                             'Entry_hour','Entry_minute','Dep_hour','Dep_minute']]        
        # 依据年、月、日、站合并
        df = pd.merge(df_IC,df_Plate,how='inner',on=['year','month','day','gas_station']) 
        print('角叉加油与号牌识别记录：{}'.format(len(df))) 
        # 筛选加油时间大于进站时间的数据集为分析数据集       
        df = df[(df.fuelle_date>df.Entry_date) & (df.shopping_date >= df.fuelle_date) & (df.shopping_date<df.Departure_date)].reset_index(drop=True)
        # 继续缩小，筛选加油时间小于离站时间的数据集为分析数据集   
        print('筛选符合时间规则记录：{}'.format(len(df)))          
        #df = df[df.shopping_date<df.Departure_date].reset_index(drop=True) 
        #print(df) 
        # 进站与出站间隔时间内，出现IC卡加油的次数，倒数为准确度的百分比
        df_combos = df.groupby(['IC_ID_R', 'License_plate_R','Entry_date', 'Departure_date' ,'gas_station'], as_index=False)['day'].count() 
        df_combos = df_combos.rename(columns={'day':'combos'})
        df = pd.merge(df,df_combos,how='inner',on=['IC_ID_R', 'License_plate_R','Entry_date', 'Departure_date','gas_station']) 
        # 增加时间间隔，加油到入站
        df['Entry_hour_interva'] = df['IC_hour'] - df['Entry_hour']
        df['Entry_minute_interva'] = df['IC_minute'] - df['Entry_minute']
        df['Shop_hour_interva'] =  df['shop_hour'] - df['IC_hour']
        df['Shop_minute_interva'] = df['shop_minute'] - df['IC_minute']               
        df['Dep_hour_interva'] =  df['Dep_hour'] - df['shop_hour']
        df['Dep_minute_interva'] = df['Dep_minute'] - df['shop_minute']       
        # 对于原始数据，如果IC_ID卡号相同，牌照号码相同，则认为真，为1，否则为错误0
        df['Flag'] = df.apply(lambda x: 1 if (x.IC_ID== x.IC_ID_R) & (x.License_plate==x.License_plate_R) else 0, axis=1)              
          
        print(df)
        #print(df_IC)  
        #print(df_Plate)
        #df['fuelle_date'] = df.apply(lambda x: datetime.datetime.strftime(x.fuelle_date,'%Y-%m-%d %H:%M:%S'))

        df['IC_time'] = df['IC_hour'] + round(df['IC_minute']/60,2)
        df['Shop_time'] = df['shop_hour'] + round(df['shop_minute']/60,2)
        df['Entry_time'] = df['Entry_hour'] + round(df['Entry_minute']/60,2)
        df['Dep_time'] = df['Dep_hour'] + round(df['Dep_minute']/60,2)
        df['Entry_interva'] = df['Entry_hour_interva'] + round(df['Entry_minute_interva']/60,2)  #替换原概念为进站到加油间隔时长
        df['Shop_interva'] = df['Shop_hour_interva'] + round(df['Shop_minute_interva']/60,2)
        df['Dep_interva'] = df['Dep_hour_interva'] + round(df['Dep_minute_interva']/60,2)
        
        # 处理过0点情况
        df['Dep_interva'] = df['Dep_interva'].apply(lambda x: x if x>=0 else (24 + x))
        df['Dep_time'] = df['Dep_time'].apply(lambda x: x if x>=1 else (24 + x))
        
        df['combos'] = 1/df['combos'] 
        df = df.drop(['Entry_hour', 'Entry_minute', 'Dep_hour', 'Dep_minute', 'Entry_hour_interva', 'Entry_minute_interva', 'Dep_hour_interva',
                          'Dep_minute_interva','Shop_hour_interva','Shop_minute_interva'],axis=1)
        
        df['fuelle_date'] = df['fuelle_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['shopping_date'] = df['shopping_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['Entry_date'] = df['Entry_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['Departure_date'] = df['Departure_date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        self.GC.generate_combo_data('combo_data', df)  
        
        return df

if __name__ == '__main__':
    
    IC_P = IC_Plate()
    
    df = IC_P.combo_data()
    print(df)
    
    pass