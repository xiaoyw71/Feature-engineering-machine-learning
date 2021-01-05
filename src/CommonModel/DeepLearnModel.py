# -*- coding: utf-8 -*-
'''
Created on 2020年12月26日

@author: 肖永威
'''
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from DataBase.Car_Info import Gas_Collection
from DataBase.DL_ComboData import ComboData

class TensorBP(object):
    def __init__(self):
        return
    
    # 输入训练数据集并设置训练与测试比例
    def set_Datas(self,x,y,test_size=0.3):
        # 归一化处理
        MM = MinMaxScaler()
        self.X_ = MM.fit_transform(x)
        self.y_= y
        #注意训练集、测试集返回参数顺序
        self.x_train,self.x_test, self.y_train, self.y_test = train_test_split(self.X_,self.y_,test_size=test_size)
                    
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
        in_units = 220  #输入23个特征，返回一个正确与错误
            
        # 定义三层BP神经网络，层数及神经元个数通过layers参数确定，两层[5,3]，只支持2或3层，其他无意义
        x,y_,loss_,train_step,correct_pred,keep_prob,accuracy,y_conv = BP_NN(in_units,layers=[160,120,2], learning_rate=0.01) #[18,10,6,2]),dropout=True)
        log_loss= []
        log_acc = []
        saver = tf.train.Saver()  #定义saver
        #随机梯度下降算法训练参数
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        
            for i in range(2000):
                batch_x,batch_y = sess.run(next_iterator)
        
                _,loss,acc = sess.run([train_step,loss_,accuracy], feed_dict={x:batch_x,y_:batch_y,keep_prob:0.8})
                
                if i%10 == 0:
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
        labels =['0','500','1000','1500','2000']
        plt.xticks(xx,labels=labels)

        plt.plot(log_loss,label = 'train-logloss',color='red')
        plt.plot(log_acc,label = 'train-correct',color='blue')
       
        plt.xlabel('训练次数')
        # 显示图例
        plt.legend()
        plt.show()

        return

class TensorLetNet5(object):
    def __init__(self):
        return

    # 输入训练数据集并设置训练与测试比例
    def set_Datas(self,x,y,test_size=0.3):
        # 归一化处理
        MM = MinMaxScaler()
        self.X_ = MM.fit_transform(x)
        self.y_= y
        #注意训练集、测试集返回参数顺序
        self.x_train,self.x_test, self.y_train, self.y_test = train_test_split(self.X_,self.y_,test_size=test_size)
                    
        return        

    def train_Model_fit(self,num_class,normalization=True):
        #标准差为0.1的正态分布
        def weight_variable(shape,name=None):
            initial = tf.truncated_normal(shape,stddev=0.1)
            return tf.Variable(initial,name=name)
        
        #0.1的偏差常数，为了避免死亡节点
        def bias_variable(shape,name=None):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial,name=name)
        
        #二维卷积函数
        #strides代表卷积模板移动的步长，全是1代表走过所有的值
        #padding设为SAME意思是保持输入输出的大小一样，使用全0补充
        def conv2d(x,W,name=None):
            return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME',name=name)
        
        #ksize [1, height, width, 1] 第一个和最后一个代表对batches和channel做池化，1代表不池化
        #strides [1, stride,stride, 1]意思是步长为2，我们使用的最大池化
        def max_pool_2x2(x,name=None):
            #return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME',name=name)
            return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME',name=name)
        #计算卷积后的图像尺寸，其中pool池化为正方形
        def figure_size(height,width,pool,stride,layer):
            for i in range(0,layer):
                height = round((height-pool)/stride) + 1
                width = round((width-pool)/stride) + 1
            return height,width
        # 定义LetNet-5神经网络，输入每组数据220个，输出2个（正确与错误），x_shape为数据形状（任意行，20*11形状，1层通道），学习率0.01
        def LetNet5(in_units=220,out_units=2,x_shape=[-1,20,11,1], learning_rate = 0.01):  #,dropout=True):       
            #x为原始输入数据，即特征向量，None代表可以批量喂入数据
            #y_为对应输入数据的期望输出结果，即真实值
            x = tf.placeholder(tf.float32, [None,in_units],name='x')
            y_ = tf.placeholder(tf.float32, [None,out_units],name='y')
            #reshape图片成28 * 28 大小，-1代表样本数量不固定，1代表channel
            x_image=tf.reshape(x,x_shape)
            height = x_shape[1]
            width = x_shape[2]
            print('height: {},width: {}'.format(height,width))
            pool = 2 #池化核2*2 ，见def max_pool_2x2(x,name=None):
            stride = 2 #步长 2
            layer = 2 #两层卷积
            
            #前面两个3代表卷积核的尺寸，1代表channel，16代表深度
            W_conv1 = weight_variable([3,3,1,16],name='w_conv1')   #16
            b_conv1 = bias_variable([16],name='b_conv1')
            h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1,name='h_conv1')
            print('h_conv1:{}'.format(h_conv1.shape))
            #第一池化层，2*2最大值池化
            p_conv1 = max_pool_2x2(h_conv1,name='pool_1') 
            print('p_conv1:{}'.format(p_conv1.shape))
            #第二卷积层的卷积核：3x3卷积核滤波，32通道，32个特征映射
            W_conv2 = weight_variable([3,3,16,32], name='w_conv2')  #3,3,16,32
            b_conv2 = bias_variable([32], name='b_conv2')
            h_conv2 = tf.nn.relu(conv2d(p_conv1,W_conv2)+b_conv2, name='h_conv2')
            print('h_conv2:{}'.format(h_conv2.shape))
            p_conv2 = max_pool_2x2(h_conv2, name='pool_2') 
            print('p_conv2:{}'.format(p_conv2.shape))
            #输入为5 * 3 * 32， 输出为480的一维向量（神经元）
            height,width = figure_size(height,width,pool,stride,layer)
            print('height: {},width: {}'.format(height,width))
            #height = 20
            #width = 10
            W_fc1 = weight_variable([height*width*32,80], name='w_fc1')
            b_fc1 = bias_variable([80], name='b_fc_1')
            #重构第二池化层，接下来要进入全连接层full_connecting
            h_pool1_flat = tf.reshape(p_conv2,[-1,height*width*32], name='pool_1_flat')
            h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat,W_fc1)+b_fc1, name='h_fc1')
            
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob, name='h_fc1_drop')
            
            #第二全连接层的权重和偏置
            W_fc2 = weight_variable([80,40], name='w_fc2')
            b_fc2 = bias_variable([40], name='b_fc_2')
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2)+b_fc2, name='h_fc2')
            
            #keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            h_fc2_drop = tf.nn.dropout(h_fc2,keep_prob, name='h_fc2_drop')

            
            W_fc3 = weight_variable([40,2], name='w_fc3')
            b_fc3 = bias_variable([2], name='b_fc3')
            #第二全连接层，即输出层，使用柔性最大值函数softmax作为激活函数
            y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop,W_fc3) + b_fc3, name='y_')
            
            # 使用TensorFlow内置的交叉熵函数避免出现权重消失问题
            cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv), name='cost_func')
            
            #使用优化器
            #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
            lr = tf.Variable(learning_rate,dtype=tf.float32)
            train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
            
            # 正确的预测结果
            correct_pred = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1), name='Correct_pred')
            # 计算预测准确率
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
            return x,y_,cross_entropy,train_step,correct_pred,keep_prob,accuracy,y_conv                  
            

        #print(inputs)
        #print(labels)
        # 定义周期、批次、数据总数、遍历一次所有数据需的迭代次数
        #n_epochs = 3
        batch_size = 20 #4
        
        # 使用from_tensor_slices将数据放入队列，使用batch和repeat划分数据批次，且让数据序列无限延续
        dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        dataset = dataset.batch(batch_size).repeat()
        
        # 使用生成器make_one_shot_iterator和get_next取数据
        iterator = dataset.make_one_shot_iterator()
        next_iterator = iterator.get_next()
        
        #定义神经网络的参数
        #in_units = 240  #输入12个特征，返回一个正确与错误
        x,y_,loss_,train_step,correct_pred,keep_prob,accuracy,y_conv = LetNet5(in_units=240,out_units=2,x_shape=[-1,20,12,1], learning_rate = 0.001) #[18,10,6,2]),dropout=True)
        log_loss= []
        log_acc = []
        saver = tf.train.Saver()  #定义saver
        #随机梯度下降算法训练参数
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        
            for i in range(2000):
                batch_x,batch_y = sess.run(next_iterator)
        
                _,loss,acc = sess.run([train_step,loss_,accuracy], feed_dict={x:batch_x,y_:batch_y,keep_prob:0.8})
                
                if i%10 == 0:
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

        #plt.figure(111)
        fig, ax1 = plt.subplots() # 使用subplots()创建窗口
        plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
        plt.grid()   
        #plt.xticks(x, labels, rotation='vertical')
        # 设置横坐标刻度标示（最大循环中是2000/10次取数据）
        xx = [0,50,100,150,200]
        labels =['0','500','1000','1500','2000']
        #ax1.xticks(xx,labels=labels)
        ax1.set_ylabel('logloss')
        ax1.set_xticks(xx)
        ax1.set_xticklabels(labels)
        ax1.plot(log_loss,label = 'train-logloss',color='red')
        ax1.set_xlabel('训练次数')
                
        ax2 = ax1.twinx() # 创建第二个坐标轴
        ax2.set_ylabel('correct')
        ax2.plot(log_acc,label = 'train-correct',color='blue')
       
        # 显示图例
        ax1.legend(loc='center')
        ax2.legend(loc='center right')
        plt.show()

        return


        

if __name__ == '__main__':  
    GC = Gas_Collection('study')

    df = GC.get_combo_data()
    
    print(df.columns)  
    # 探索性数据分析
    CbD = ComboData(df)
    
    #EDA.correlation_analysis(cols_name)
    CbD.set_datas(times=1)
    x,y = CbD.get_datas()
    
    #TB = TensorBP()
    TB = TensorLetNet5()
    TB.set_Datas(x,y, 0.3)
    TB.train_Model_fit(2, normalization=False)
    
        
    pass        
        