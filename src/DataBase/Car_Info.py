# -*- coding: utf-8 -*-
'''
Created on 2020年12月16日

@author: 肖永威
'''
import pymongo
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import datetime
import math

#车与IC卡信息类
class Car_IC_Info(object):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor

        '''    
    # 按整车装备质量仿真油耗，油箱容积、日均里程
    def fuel_behavior(self,num=1000,IC_Num=500):
        fc = [] #油耗
        full_mass = [] # 整车质量
        ic_id = [] #IC卡号，大于1000的为非IC加油数据，混入非IC卡车牌
        c_id = [] # 客户号
        standard_vol = [35,45,50,55,60,65,70,75,80,90,100] #标准油箱容积
        vol = [] #油箱容积
        creation = ['2019-10-30 00:00:00','2019-10-25 00:00:00','2019-11-03 00:00:00','2019-10-20 00:00:00','2019-11-01 00:00:00']
        creation_date = [] #创建日期
        License_plate = [] #车牌
        datas = [] # 数据集
        fuel_datetime = [5,7,8,10,12,13,14,16,18,19,20,21]
        behavior_datetime = [] #加油习惯时间
        behavior = [] #加油行为习惯
        behavior_shoping = [0,0,1]

        station = [123,132,213,231,312,321] 
        behavior_station = [] #习惯加油地点，占比[0.6,0.3,0.1]
        # 取标准油箱容积
        def get_standard_vol(sv,v):
            sv = np.array(sv)
            sv = sv[sv>v]
            k = np.argmin(sv)
                        
            return int(sv[k])
        
        for i in range(num):
            fm = int(random.normalvariate(mu=1500,sigma=200)/100)*100
            full_mass.append(fm)
            c = round(random.normalvariate(mu=8.0*fm/1500,sigma=0.6),1)
            v = get_standard_vol(standard_vol,c*random.randint(500,650)/100)
            vol.append(v)
            m0 = float(v)/c
            # 处理油耗过低的情况
            if m0>6.3:
                c = c + round(math.sqrt(m0 - 6.3),1) #如果按油耗计算，行驶里程大于630公里，则油耗+差开根号
            fc.append(c)
            creation_date.append(creation[random.randint(0,4)])
            behavior_datetime.append(fuel_datetime[random.randint(0,11)])
            behavior_station.append(station[random.randint(0,5)])
            behavior.append(random.randint(0,5))
            
            c_id.append(i+1)
            if i<IC_Num:
                ic_id.append(i+1)
            else:
                ic_id.append(-1) #无卡用户
            License_plate.append(900000+i)
                 
        # 截断正态分布
        mu, sigma = 42, 70
        lower, upper = mu - 1/3 * sigma, mu + 3 * sigma  # 截断在[μ-1/3σ, μ+3σ]
        #日均里程
        mileage=(stats.truncnorm((lower - mu)/sigma, (upper - mu)/sigma, loc=mu, scale=sigma)).rvs(num) #有区间限制的随机数(mu=35,sigma=10)) 
        mileage= np.around(mileage, 0)  
        for i in range(num):       
            data_dict = {'C_ID':0,'IC_ID':0,'License_plate':0,'price_sensitive':0,'vol':0,'mileage':0,'fuel_consumption':0,'full_mass':0,
                         'creation':'2019-12-30','gas_station':0,'behavior_datetime':0,'behavior_week':0,'behavior_station':0,'behavior_shoping':0,'behavior':0}
            data_dict['IC_ID'] = ic_id[i]
            data_dict['C_ID'] = c_id[i]
            data_dict['License_plate'] = License_plate[i]
            data_dict['price_sensitive'] = random.randint(0,2)
            data_dict['vol'] = vol[i]
            data_dict['mileage'] = mileage[i]
            data_dict['full_mass'] = full_mass[i]
            data_dict['gas_station'] = random.randint(0,2)
            data_dict['fuel_consumption'] = fc[i]
            data_dict['creation'] = creation_date[i]
            data_dict['behavior_datetime'] = behavior_datetime[i]
            data_dict['behavior_station'] = behavior_station[i]
            data_dict['behavior_week'] = random.randint(0,1)  # 0是工作日，1是周末加油
            data_dict['behavior_shoping'] = behavior_shoping[random.randint(0,2)]  # 0不是 1是习惯购物，存在多次刷卡情况，停车时间长
            data_dict['behavior'] = behavior[i]

            datas.append(data_dict)                     
          
        #plt.hist(vol) #画密度图
        #plt.show()
        '''
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        colors = '#DC143C' #点的颜色
        plt.xlabel('整备质量')
        plt.ylabel('油耗')

        area = np.pi * 3**2  # 点面积 
        # 画散点图
        plt.scatter(full_mass, fc, s=area, c=colors, alpha=0.4)
        plt.show()
        '''    
        return datas
    
    #预测加油日期
    '''
    initial，油箱初始化，或者加完油后
    '''
    def get_fuelle_date(self,fuelle_date,fuel_consumption,mileage,vol,Remaining_oil,Price_sensitive,price,behavior_week,behavior_datetime):
        # 计算一次加油后，大致运行天数，油箱剩余油按10%浮动，油耗按5%浮动
        remain = Remaining_oil * (1 + random.uniform(-0.1,0.1)) #剩余油  
        fuelle = (vol - remain)*(1-random.uniform(-0.1,0.2)) #加油量
        
        date_p = datetime.datetime.strptime(fuelle_date,'%Y-%m-%d %H:%M:%S')
        date_str = date_p.strftime('%Y-%m-%d %H:%M:%S')
        date_str = date_str[0:10] + ' 00:00:00'
        date_p = datetime.datetime.strptime(date_str,'%Y-%m-%d %H:%M:%S')
        days = fuelle/(mileage*fuel_consumption/100*((1 + random.uniform(-0.05,0.05))))
        if days < 1:
            days = 1
        days = int(days)
        #临时预算加油日期及周几
        date_p0 = date_p + datetime.timedelta(days=days)
        weeks = date_p0.weekday()
        # 周末加油习惯处理
        if behavior_week>0:
            if weeks < 5:
                # 周一、二，提前加油
                if weeks <= 1 and days > 2:
                    days = days - weeks - 1
                    remain = remain + (mileage*fuel_consumption/100)*(weeks + 1)
                    fuelle = (vol - remain)*(1-random.uniform(-0.1,0.2)) #加油量  
                # 周五滞后一天加油
                if weeks ==4 and remain/(mileage*fuel_consumption/100)>1:
                    days = days + 1
                    remain = remain - (mileage*fuel_consumption/100)
                    fuelle = (vol - remain)*(1-random.uniform(-0.1,0.2)) #加油量  
        #限定天数，重新计算加油量              
        amount = fuelle*price  
        if random.randint(1,10) > 2: #向上取整
            amount = math.ceil(amount/10)*10
        else:
            amount = round(amount,0)
        fuelle = amount/price
        date_p = date_p + datetime.timedelta(days=days) 
        # 加油时间行为习惯
        date_p = date_p + datetime.timedelta(hours=behavior_datetime) + datetime.timedelta(hours=random.randint(-1,1))
        date_p = date_p + datetime.timedelta(minutes=random.randint(-30,30))        
           
        #返回加油日期、加油量
        return fuelle_date, date_p.strftime('%Y-%m-%d %H:%M:%S'), round(fuelle,2), amount, days
    
    #  取当前油价
    def get_price(self,df_price,fuelle_date):
        #fuelle_date = datetime.datetime.strptime(fuelle_date,'%Y-%m-%d %H:%M:%S').date()
        row = df_price.loc[(df_price['dt']<=fuelle_date)].sort_values(by='dt', ascending=False).iloc[0]
    
        dt = row['dt'] #.strftime('%Y-%m-%d %H:%M:%S')
        price = row['price']
        changes = row['price']
        
        return dt, price, changes
    
    #涨价提前加油，
    def get_rise_fuelle_date(self,fuelle_date,fuel_consumption,mileage,vol,rise_date,Remaining_oil,initial,Price_sensitive,price,behavior_datetime):
        date_p = datetime.datetime.strptime(fuelle_date,'%Y-%m-%d %H:%M:%S')
        date_str = date_p.strftime('%Y-%m-%d %H:%M:%S')
        date_str = date_str[0:10] + ' 00:00:00'
        date_p = datetime.datetime.strptime(date_str,'%Y-%m-%d %H:%M:%S')
        date_e = datetime.datetime.strptime(rise_date,'%Y-%m-%d %H:%M:%S') - datetime.timedelta(days=1)
        days = (date_e - date_p).days
        remain = Remaining_oil * (1 + random.uniform(-0.1,0.1)) #前期理论剩余油 
        remain =  (initial - days*mileage*fuel_consumption/100*((1 + random.uniform(-0.05,0.05)))) + remain #当前剩余油
        fuelle = (vol - remain)*(1-random.uniform(0.1,0.2)) #加满油，加油量
        
        amount = round(fuelle*price,2)  
        if random.randint(1,10) > 2: #向上取整
            amount = math.ceil(amount/10)*10
        else:
            amount = round(amount,0)
        fuelle = amount/price
        days = (vol - Remaining_oil * (1 + random.uniform(-0.1,0.1)))/(mileage*fuel_consumption/100*((1 + random.uniform(-0.05,0.05))))      
        days = math.ceil(days)
        
        if days < 1:
            days = 1
        days = int(days)
        date_p = date_e + datetime.timedelta(days=days)  
        # 加油时间行为习惯
        date_p = date_p + datetime.timedelta(hours=behavior_datetime) + datetime.timedelta(hours=random.randint(-1,1))
        date_p = date_p + datetime.timedelta(minutes=random.randint(-30,30))  #可以考虑正态分布
        
        return date_e.strftime('%Y-%m-%d %H:%M:%S'), date_p.strftime('%Y-%m-%d %H:%M:%S'), round(fuelle,2), amount, days
        
    #随机加油记录，LP为号牌识别类对象
    def generate_gas_sell_record(self,df_IC_Info,df_gas_price, LP,CS):
        df_gas_member = df_IC_Info
        df_price = df_gas_price
        df_sales = pd.DataFrame(columns=('C_ID','IC_ID','License_plate','fuelle_date','price','fuelle','amount','Payment','gas_station'))

        # 取随机加油站
        def get_behavior_station(behavior_station):
            per = random.randint(0,9)
            k = 0
            if per >=9 :
                k = 2
            elif per >=6:
                k= 1
                
            behavior_station = str(behavior_station) 
            
            return int(behavior_station[k])
        num = len(df_gas_member)
        for index,row in df_gas_member.iterrows():
            ic_id = row['IC_ID']
            c_id = row['C_ID']
            License_plate = row['License_plate']
            ic_id0 = ic_id

            vol = row['vol']
            mileage = row['mileage']
            Remaining_oil = 5*(1 + random.randint(-10,10)/100)
            initial = 5*(1 + random.randint(-10,10)/100)
            fuel_consumption = row['fuel_consumption']
            creation = row['creation']
            gas_station = row['gas_station']
            behavior_station = row['behavior_station']
            
            Price_sensitive = row['price_sensitive']
            fuelle_date = creation
            Payment = 5 # random.randint(1,5) # 1现金 2 微信 3 支付宝 4 银行卡 5加油卡
            price_base = 6.77
            days = 0
            label_price = True  # 标识涨价头一天加油
            fuelle_date_end = creation
            
            behavior_week = row['behavior_week']
            behavior_datetime = row['behavior_datetime']
            behavior = row['behavior']
            # 设置配套车辆号牌识别记录
            LP.set_params(c_id, ic_id, License_plate, behavior)
            # 设置非油品购物操作
            behavior_shoping = row['behavior_shoping']

            CS.set_shopping_params(c_id, ic_id, License_plate,behavior_shoping)
            
            while fuelle_date<='2020-12-15':
                if price_base > 6.0:
                    if Price_sensitive == 2:
                        mileage = mileage * random.uniform(0.8,1.0)
                        fuel_consumption = fuel_consumption * random.uniform(0.95,1.0)
                    elif Price_sensitive == 1:
                        mileage = mileage * random.uniform(0.9,1.0)
                elif price_base<5.5:
                    if Price_sensitive == 2:
                        mileage = mileage * random.uniform(1.0,1.2)
                        fuel_consumption = fuel_consumption * random.uniform(1.0,1.05)
                    elif Price_sensitive == 1:
                        mileage = mileage * random.uniform(1.0,1.1)
                # 行驶里程，随机浮动
                mileage = mileage * random.uniform(0.9,1.1)
                # 随机获取加油站加油
                station = get_behavior_station(behavior_station)
                # 获取当前油价
                #rise_date, price ,changes = get_price(df_price,fuelle_date) 
                rise_date, next_price ,changes = self.get_price(df_price,fuelle_date_end)
                
                amount = 0.00                           
                if next_price > price_base and days > 1 and label_price: # and interva == 0:
                    label_price = False

                    if Price_sensitive == 2: # 价格敏感型
                        fuelle_date, fuelle_date_end, fuelle, amount, days = self.get_rise_fuelle_date(fuelle_date,fuel_consumption,mileage,vol,rise_date,Remaining_oil*(1++random.uniform(-0.1,0.1)),initial,Price_sensitive,price_base,behavior_datetime)
                    elif Price_sensitive == 1:
                        if random.randint(1,10)>=4: # 价格敏感度不太高的，按60%随机处理
                            fuelle_date, fuelle_date_end, fuelle, amount, days = self.get_rise_fuelle_date(fuelle_date,fuel_consumption,mileage,vol,rise_date,Remaining_oil*(1++random.uniform(-0.1,0.1)),initial,Price_sensitive,price_base,behavior_datetime)
                        else:
                            fuelle_date, fuelle_date_end, fuelle, amount, days = self.get_fuelle_date(fuelle_date_end,fuel_consumption,mileage,vol,Remaining_oil*(1++random.uniform(-0.1,0.1)),Price_sensitive,next_price,behavior_week,behavior_datetime)
                            # 虽然涨价，未涨价前加油
                            price_base = next_price
                            label_price = True
                    else:
                        fuelle_date, fuelle_date_end, fuelle, amount, days = self.get_fuelle_date(fuelle_date_end,fuel_consumption,mileage,vol,Remaining_oil*(1++random.uniform(-0.1,0.1)),Price_sensitive,next_price,behavior_week,behavior_datetime)
                        # 虽然涨价，未涨价前加油
                        price_base = next_price
                        label_price = True                    
    
                else:
                    fuelle_date, fuelle_date_end, fuelle, amount, days = self.get_fuelle_date(fuelle_date_end,fuel_consumption,mileage,vol,Remaining_oil*(1++random.uniform(-0.1,0.1)),Price_sensitive,
                                                                                              price_base,behavior_week,behavior_datetime)
                    label_price = True       
                
                initial = fuelle
                # 存在不用卡加油的情况，10%比例算
                if random.randint(0,10)==9:
                    ic_id = -1 
                else:
                    ic_id =ic_id0  
    
                #存在10%在其他地方加油的情况，直接越过加油记录，体现不连续
                #购物时间，默认为0
                #shop_time = 0
                shop_time = CS.generate_Shopping_Record(fuelle_date,station)
                # 产生号牌识别记录
                LP.generate_records_params(fuelle_date,fuelle,station,shop_time)
                # 随机产生识别号牌记录
                #if random.randint(0,9)<7:
                #    LP.generate_records_random(fuelle_date,station)
                if random.randint(0,9)<9:
                    df_sales = df_sales.append(pd.DataFrame({'C_ID':[c_id],'IC_ID':[ic_id],'License_plate':[License_plate],'fuelle_date':[fuelle_date],'price':[price_base],'fuelle':[fuelle],'amount':[amount],
                                              'Payment':[Payment],'gas_station':[station]}), ignore_index=True)
                price_base = next_price  
            
            if index%10==0:
                print('完成：{:%}，{}'.format(round(index/num,4),index))               
        
        return df_sales        

# 车牌识别系统信息
class License_Plate(object):
    def __init__(self):
        # IC卡ID，车牌号，进入时间，离开时间，加油站
        self.df_Plate_Record = pd.DataFrame(columns=('C_ID','IC_ID','License_plate','Entry_date','Departure_date','gas_station'))
        self.Entry_time_list = [3,4,7,7,5,5]
        self.Departure_time_list = [3,4,5,6,4,5]
        self.fluctuate_list = [0.3,0.5,0.2,0.3,0.1,0.5] 
        
        return
    # 设置号牌行为特征参数
    def set_params(self,C_ID,IC_ID,License_ID,behavior):
        self.c_id = C_ID
        self.IC_ID = IC_ID
        self.behsvior = behavior
        self.License_ID = License_ID
        self.Entry_time = self.Entry_time_list[behavior]
        self.Departure_time = self.Departure_time_list[behavior]*self.fluctuate_list[behavior]
        self.fluctuate0 = 1.0 - self.fluctuate_list[behavior] #行为波动时长比例低端
        self.fluctuate1 = 1.0 + self.fluctuate_list[behavior] #行为波动时长比例高端
                
        return
    # 设置号牌识别参数，加油时间、加油量（用于计算加油时间）、加油站
    # 增加非油购物消耗时间shop_time
    def generate_records_params(self,fuelle_date,fuelle,station,shop_time):
        velocity = 37 # 标准流速
        date_p = datetime.datetime.strptime(fuelle_date,'%Y-%m-%d %H:%M:%S')
        
        Entry_date = date_p + datetime.timedelta(minutes=-(self.Entry_time*random.uniform(self.fluctuate0,self.fluctuate1))) 
        # 离开时间 = 进入时间+加油时长+离开时长+非油购物时长
        Departure_date = date_p + datetime.timedelta(minutes=(round(fuelle/velocity,1) + 
                                self.Departure_time*random.uniform(self.fluctuate0,self.fluctuate1) + shop_time)) 
        # 可能未识别号牌的情况
        if random.randint(0,10)<10:
            self.df_Plate_Record = self.df_Plate_Record.append(pd.DataFrame({'C_ID':[self.c_id],'IC_ID':[self.IC_ID],'License_plate':[self.License_ID],'Entry_date':[Entry_date.strftime('%Y-%m-%d %H:%M:%S')],
                                        'Departure_date':[Departure_date.strftime('%Y-%m-%d %H:%M:%S')],'gas_station':[station]}), ignore_index=True)
        
        return self.df_Plate_Record
    #随机产生号牌识别记录，干扰数据
    def generate_records_random(self,fuelle_date,station):
        date_p = datetime.datetime.strptime(fuelle_date,'%Y-%m-%d %H:%M:%S')
        
        Entry_date = date_p + datetime.timedelta(minutes=-random.uniform(60,70)) 
        
        Departure_date = date_p + datetime.timedelta(minutes=random.uniform(10,70)) 
        self.df_Plate_Record = self.df_Plate_Record.append(pd.DataFrame({'C_ID':[self.c_id],'IC_ID':-2,'License_plate':[800000+random.randint(1,1000)],'Entry_date':[Entry_date.strftime('%Y-%m-%d %H:%M:%S')],
                                        'Departure_date':[Departure_date.strftime('%Y-%m-%d %H:%M:%S')],'gas_station':[station]}), ignore_index=True)
        
        return self.df_Plate_Record
    
    def get_Plate_Record(self):
        return self.df_Plate_Record    
        
    def generate_License_Plate_record(self,df_IC_sales):
        # 识别车牌与加油间隔时间分布
        mu, sigma = 2,2
        SAMPLE_SIZE = 1000
        # 伽玛分布
        interva = [random.gammavariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)]
        plt.hist(interva) #画密度图
        plt.show()  
               
        mu = 3        
        loc = 1
        #泊松分布
        interva = stats.poisson.rvs(mu,loc,SAMPLE_SIZE)
        
        plt.hist(interva) #画密度图
        plt.show()
        # 有识别号牌而不加油的情况
        '''
        for index,row in df_IC_sales.iterrows():
            fuel_time = row['fuelle_date']
            License_plate = row['License_plate']
            _id = row['License_plate']
            gas_station = row['gas_station']
            
            #Entry_time = 
            #Departure_time
        ''' 
                        
        return

# IC非油购物
class Car_Shopping(object):
    def __init__(self):
        # IC卡ID，车牌号，刷卡时间，购物金额
        self.df_Shopping_Record = pd.DataFrame(columns=('C_ID','IC_ID','License_plate','shopping_date','amount','gas_station'))
        self.shopping_time_list = [5,8,10,20,30]
        
        return
        
    # 设置消费行为特征参数
    def set_shopping_params(self,C_ID,IC_ID,License_ID,behavior_shoping):
        self.c_id = C_ID
        self.IC_ID = IC_ID
        self.License_ID = License_ID
        self.shopping_time = self.shopping_time_list[random.randint(0,4)]*random.uniform(0.8,1.2)
        self.behavior_shoping = behavior_shoping        
        return
    # 设置非油购物记录
    def generate_Shopping_Record(self,fuelle_date,station):
        Shopping = False
        shop_time = 0
        if self.behavior_shoping == 1:
            # 80%情况下购物
            if random.randint(0,9)>1:            
                Shopping = True
                shop_time = self.shopping_time*random.uniform(0.8,1.2)
        else:
            # 随机给卡充值，1/10频率
            if random.randint(0,9)==0:
                Shopping = True
                shop_time = 8*random.uniform(0.8,1.5)
            else:
                # 偶尔非油消费购物
                if random.randint(0,20)==0:
                    Shopping = True
                    shop_time = 5*random.uniform(0.8,1.5)                
            
        if Shopping:                    
            date_p = datetime.datetime.strptime(fuelle_date,'%Y-%m-%d %H:%M:%S')
            
            Shopping_date = date_p + datetime.timedelta(minutes=shop_time) 
            
            self.df_Shopping_Record = self.df_Shopping_Record.append(pd.DataFrame({'C_ID':[self.c_id],'IC_ID':[self.IC_ID],'License_plate':[self.License_ID],
                                            'shopping_date':[Shopping_date.strftime('%Y-%m-%d %H:%M:%S')],
                                            'amount':[random.uniform(1.00,1000.00)],'gas_station':[station]}), ignore_index=True)
        
        return shop_time

    
    def get_Shopping_Record(self):
        return self.df_Shopping_Record    
        
        
#数据集类          
class Gas_Collection(object):
    
    def create_collection(self,name,data):
        self.collection = self.db[name]
        self.collection.insert_many(data)
        return

    def __init__(self, db_name):
        '''
        Constructor
        '''
        self.db_name = db_name
        client = pymongo.MongoClient('mongodb://study:study@localhost:27017/study')
        self.db = client[self.db_name]
    # 取IC卡信息    
    def get_IC_Info(self):
        collection = self.db['IC_Info']
   
        df = pd.DataFrame(list(collection.find()))
        
        return df
    # 取油价列表
    def get_gas_price(self): 
        collection = self.db['gas_price']
   
        df = pd.DataFrame(list(collection.find()))
        df['changes'] = df['changes'].astype('float')
        
        return df
    # 取IC卡加油信息
    def get_IC_Sales(self):
        collection = self.db['IC_sales']
   
        df = pd.DataFrame(list(collection.find()))
        
        return df        
            
    #产生IC卡加油信息
    def generate_sales(self,name,datas):
        collection = self.db[name]   
        
        collection.insert(json.loads(datas.T.to_json()).values())
        
        return

    #产生号牌识别信息
    def generate_License(self,name,datas):
        collection = self.db[name]   
        
        collection.insert(json.loads(datas.T.to_json()).values())
        
        return
    #产生非油购物记录
    def generate_Shopping(self,name,datas):
        collection = self.db[name]   
        
        collection.insert(json.loads(datas.T.to_json()).values())
        
        return
    #产生IC卡与车号牌组合初级数据集
    def generate_combo_data(self,name,datas):
        collection = self.db[name]   
        
        collection.insert(json.loads(datas.T.to_json()).values())
        
        return
    # 取车牌识别信息
    def get_Plate_record(self): 
        collection = self.db['Plate_record']
   
        df = pd.DataFrame(list(collection.find()))
        
        return df
    # 取非油购物数据
    def get_Shopping_data(self):
        collection = self.db['Shopping_record']
   
        df = pd.DataFrame(list(collection.find()))
        
        return df      
    
    # 取初步组合数据
    def get_combo_data(self):
        collection = self.db['combo_data']
   
        df = pd.DataFrame(list(collection.find()))
        
        return df  
    #产生分析数据集
    def generate_Analysis_data(self,name,datas):
        collection = self.db[name]   
        
        collection.insert(json.loads(datas.T.to_json()).values())
        
        return

    #提取分析数据集
    def get_Analysis_data(self,dbname,cols_name):
        collection = self.db[dbname]   
        
        df = pd.DataFrame(list(collection.find()))
        
        return df[cols_name]
        
def generate_collection():
    Car = Car_IC_Info()
    data = Car.fuel_behavior(2500,1100)
    IC_Info = Gas_Collection('study')
    IC_Info.create_collection('IC_Info', data)

    return

def generate_IC_sales():
    Car = Car_IC_Info()
    LP= License_Plate()
    C_Shop = Car_Shopping()
    
    IC_Info = Gas_Collection('study')
    df_IC_info = IC_Info.get_IC_Info()
    df_price = IC_Info.get_gas_price()
    data = Car.generate_gas_sell_record(df_IC_info, df_price,LP,C_Shop)
    IC_Info.generate_sales('IC_sales', data)
    
    plate_data = LP.get_Plate_Record()
    IC_Info.generate_License('Plate_record', plate_data)
    
    Shopping_data = C_Shop.get_Shopping_Record()
    IC_Info.generate_Shopping('Shopping_record', Shopping_data)

    return

if __name__ == '__main__':
     
    generate_collection()
    generate_IC_sales()
    #LP = License_Plate()
    #LP.generate_License_Plate_record()
            