# -*- coding: utf-8 -*-
'''
Created on 2020年12月26日

@author: Administrator
'''
import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from DataBase.Car_Info import Gas_Collection

class ClassficationXGBoost(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    # 输入训练数据集并设置训练与测试比例
    def set_Datas(self,datas,y_cols_name,test_size=0.3):
        self.datas = datas
        self.y_cols_name = y_cols_name
        self.y_ = datas[[y_cols_name]]
        self.X_ = datas.drop(y_cols_name,axis=1)  
        #注意训练集、测试集返回参数顺序
        self.x_train,self.x_test, self.y_train, self.y_test = train_test_split(self.X_,self.y_,test_size=test_size)
             
        return
    
    def train_Model_fit(self,num_class,normalization=True):
        params ={'learning_rate': 0.05,
          'max_depth': 5,                # 构建树的深度，越大越容易过拟合
          'num_boost_round':100,
          #'objective': 'multi:softprob', # 多分类的问题
          'objective': 'binary:logistic', # 二分类：binary:logistic
          #'objective': 'binary:logitraw',  # 二分类
          'random_state': 7,
          'silent':0,
          'subsample':0.9,
          'min_child_weight':5,
          #'num_class':2,                 # 类别数，与 multisoftmax 并用
          #'eval_metric':['mlogloss','merror','auc'],   # 多分类情况
          'eval_metric':['logloss','error','auc'], # 二分类情况
          'eta':0.3                      #为了防止过拟合，更新过程中用到的收缩步长。eta通过缩减特征 的权重使提升计算过程更加保守。缺省值为0.3，取值范围为：[0,1]
        }
        dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
        dtest = xgb.DMatrix(self.x_text,label=self.y_test)
        
        res = xgb.cv(params,dtrain) #,num_boost_round=5000,metrics='auc',early_stopping_rounds=25)
        #找到最佳迭代轮数
        best_nround = res.shape[0] - 1
        
        if best_nround<10:
            best_nround = 10
        
        watchlist = [(dtrain,'train'),(dtest,'eval')]
        evals_result = {}
        
        model = xgb.train(params,dtrain,num_boost_round=best_nround,evals = watchlist,evals_result=evals_result)
        y_pred=model.predict(xgb.DMatrix(self.x_test))        
        
        model.save_model('XGboostClass.model')
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
        plt.show()
        #写日志
        with open('train.log','a+',encoding='utf-8') as fw:
            nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rowline = nowTime + "\n"
            rowline = rowline + str(params) + "\n"
            rowline = rowline + "Accuracy: %.2f%%" % (accuracy * 100.0) + "\n"
            
            fw.write(rowline) 

# https://xgboost.readthedocs.io/en/latest/python/python_api.html

class ClassficationTensorBP(object):
    def __init__(self):
        return
    
    # 输入训练数据集并设置训练与测试比例
    def set_Datas(self,datas,y_cols_name,test_size=0.3):
        self.datas = datas
        self.y_cols_name = y_cols_name
        y_df = datas[[y_cols_name]]
        
        self.y_=pd.DataFrame(columns=('TrueIC','FalseIC'))
        self.y_['TrueIC']=y_df['Flag'].apply(lambda x:0 if x==0 else 1)
        self.y_['FalseIC']=y_df['Flag'].apply(lambda x:1 if x==0 else 0)
        self.X_ = datas.drop(y_cols_name,axis=1)  
        # 归一化处理
        MM = MinMaxScaler()
        self.X_ = MM.fit_transform(self.X_.values )
        self.y_= self.y_.values
        #注意训练集、测试集返回参数顺序
        self.x_train,self.x_test, self.y_train, self.y_test = train_test_split(self.X_,self.y_,test_size=test_size)
        
        #self.x_train = self.x_train.values
        #self.x_test = self.x_test.values
        #self.y_train= self.y_train.values
        #self.y_test = self.y_test.values
             
        return    

    def train_Model_fit(self,num_class,normalization=True):
        #定义神经元
        def NN(h_in,h_out,layer='1'):
            w = tf.Variable(tf.truncated_normal([h_in,h_out],stddev=0.1),name='weights' +layer )
            b = tf.Variable(tf.zeros([h_out],dtype=tf.float32),name='biases' + layer)
            
            return w,b
        
        #定义BP神经网络
        def BP_NN(in_units,layers=[18,10,6,2], learning_rate = 0.01):  #,dropout=True):
            #定义输入变量
            num = len(layers)   # 网络层数            
            x = tf.placeholder(dtype=tf.float32,shape=[None,in_units],name='x')
            #定义输出变量    
            y_ = tf.placeholder(dtype=tf.float32,shape=[None,layers[num - 1]],name='y_')
            
            #定义网络参数
            w1,b1 = NN(in_units,layers[0],'1')   #定义第一层参数
            #定义网络隐藏层
            #定义前向传播过程
            h1 = tf.nn.relu(tf.add(tf.matmul(x,w1),b1))
            #定义dropout保留的节点数量
            keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')
  
            #使用dropout
            h1_drop = tf.nn.dropout(h1,rate = 1 - keep_prob)        #Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
            
            w2,b2 = NN(layers[0],layers[1],'2')   #定义第二层参数
            # 定义第二层隐藏层
            h2 = tf.nn.relu(tf.add(tf.matmul(h1_drop,w2),b2))

            w3,b3 = NN(layers[1],layers[2],'3')   #定义第三层参数
            #h3 = tf.nn.relu(tf.add(tf.matmul(h2,w3),b3))   
            y_conv = tf.nn.softmax(tf.add(tf.matmul(h2,w3),b3),name='y_conv')         
            
            #w4,b4 = NN(layers[2],layers[3],'4')   #定义第四层参数
            # 定义输出层
            #y_conv = tf.nn.softmax(tf.add(tf.matmul(h3,w4),b4),name='y_conv') 


            #定义损失函数
            loss_func = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
            #学习率
            #learning_rate = #0.01
            lr = tf.Variable(learning_rate,dtype=tf.float32)
            train_step = tf.train.AdagradOptimizer(lr).minimize(loss_func)
            
            correct_pred = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            
            return x,y_,loss_func,train_step,correct_pred,keep_prob,accuracy,y_conv        

        #print(inputs)
        #print(labels)
        # 定义周期、批次、数据总数、遍历一次所有数据需的迭代次数
        n_epochs = 3
        batch_size = 20 #4
        
        # 使用from_tensor_slices将数据放入队列，使用batch和repeat划分数据批次，且让数据序列无限延续
        dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        dataset = dataset.batch(batch_size).repeat()
        
        # 使用生成器make_one_shot_iterator和get_next取数据
        iterator = dataset.make_one_shot_iterator()
        next_iterator = iterator.get_next()
        
        #定义神经网络的参数
        in_units = 23  #输入23个特征，返回一个正确与错误
            
        # 定义三层BP神经网络，层数及神经元个数通过layers参数确定，两层[5,3]，只支持2或3层，其他无意义
        x,y_,loss_,train_step,correct_pred,keep_prob,accuracy,y_conv = BP_NN(in_units,layers=[16,12,2], learning_rate=0.01) #[18,10,6,2]),dropout=True)
        log_loss= []
        log_acc = []
        saver = tf.train.Saver()  #定义saver
        #随机梯度下降算法训练参数
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        
            for i in range(20000):
                batch_x,batch_y = sess.run(next_iterator)
        
                _,loss,acc = sess.run([train_step,loss_,accuracy], feed_dict={x:batch_x,y_:batch_y,keep_prob:0.8})
                
                if i%100 == 0:
                    #train_accuracy = accuracy.eval(session = sess,feed_dict={x:batch_x, y_: batch_y, keep_prob: 1.0}) # 用于分类识别，判断准确率
                    #print ("step {}, training accuracy {}".format(i, train_accuracy))
                    #print ("step： {}, total_loss： {},accuracy： {},train_accuracy：{}".format(i, total_loss,accuracy, train_accuracy))           # 用于趋势回归，预测值
                    print ("step： {}, total_loss： {},accuracy： {}".format(i, loss,acc))           # 用于趋势回归，预测值
                    log_loss.append(loss)
                    log_acc.append(acc)
                    
                
            #saver.save(sess, 'save_m/BP_model.ckpt') #模型储存位置
            
            # 训练集验证
            train_ret = sess.run(y_conv, feed_dict={x:self.x_train,keep_prob:1.0})
            train_y = sess.run(tf.argmax(train_ret,1))  # 用于分类问题，取最大概率，返回训练验证结果
            y_train = sess.run(tf.argmax(self.y_train,1))  # 用于分类问题，取最大概率，还原实际数据集（oneHot编码还原）
            train_correct = sess.run(tf.equal(train_y,y_train)) #测试集结果对比（真/假）           


            # 测试集验证
            ret = sess.run(y_conv, feed_dict={x:self.x_test,keep_prob:1.0})
            y = sess.run(tf.argmax(ret,1))  # 用于分类问题，取最大概率，返回测试集验证结果
            y_test = sess.run(tf.argmax(self.y_test,1))  # 用于分类问题，取最大概率 ，还原实际数据集（oneHot编码还原）           
            test_correct = sess.run(tf.equal(y,y_test)) #测试集结果对比（真/假）           
            
            print("测试集预测结果概率：{}".format(ret))
            print("测试集实际数据：{}".format(self.y_test))           
            print("预测结果：{}".format(y))            
            print('测试集：{}'.format(y_test))
            print('测试集准确率：{}'.format(sess.run(tf.reduce_mean(tf.cast(test_correct, tf.float32)))))
            print('训练集准确率：{}'.format(sess.run(tf.reduce_mean(tf.cast(train_correct, tf.float32)))))

        plt.figure(111)
        plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
        plt.grid()   
        #plt.xticks(x, labels, rotation='vertical')
        # 设置横坐标刻度标示（最大循环中是20000/100次取数据）
        xx = [0,50,100,150,200]
        labels =['0','5000','10000','15000','20000']
        plt.xticks(xx,labels=labels)

        plt.plot(log_loss,label = 'train-logloss',color='red')
        plt.plot(log_acc,label = 'train-correct',color='blue')
       
        plt.xlabel('训练次数')
        # 显示图例
        plt.legend()
        plt.show()

        return



        

if __name__ == '__main__':  
    GC = Gas_Collection('study')
    cols_name = ['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R',
                    'Flag', 'times', 'fuel_time', 'IC_time', 'Shop_time', 'Entry_time',
                    'Dep_time', 'Entry_interva', 'Shop_interva', 'Dep_interva', 'combos',
                    'Plate_interva', 'Entry_pre', 'Dep_pre', 'fuel_time_std', 'IC_time_std',
                    'Shop_time_std', 'Entry_time_std', 'Dep_time_std', 'Entry_interva_std',
                    'Shop_interva_std', 'Dep_interva_std', 'Plate_interva_std',
                    'Entry_pre_std', 'Dep_pre_std']
    datas = GC.get_Analysis_data('Analysis_data1', cols_name)    
    
    cols_name = ['Flag', 'times','fuel_time','IC_time','Entry_time','Dep_time', 'Entry_interva', 'Dep_interva','combos',
                            'fuel_time_std','IC_time_std','Entry_time_std','Dep_time_std', 'Entry_interva_std', 'Dep_interva_std',
                            'Plate_interva','Entry_pre','Dep_pre','Plate_interva_std','Entry_pre_std','Dep_pre_std','Shop_time',
                            'Shop_interva','Shop_interva_std']    
    df = datas[cols_name]
    
    #CX = ClassficationXGBoost()
    CX = ClassficationTensorBP()
    CX.set_Datas(df, 'Flag', 0.3)
    CX.train_Model_fit(2, normalization=False)
    
        
    pass        
        