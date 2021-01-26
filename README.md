机器学习、深度学习算法是如何在软件开发过程中应用的？大数据人工智能开发过程又是什么样的呢？大数据人工智能技术能为业务带来什么呢？

这些问题往往困扰我周围很多IT开发人员、产品设计人员，新技术不仅仅是用来解决问题的，更是创新引领业务发展的。

其实，这些事需要我们建立新的结构化思维体系，通过构建产品化分层次模型和开发过程模型，落地我们的研发课题。

# 1. 前言
我们先从金字塔原理开始：结论先行、以上统下、归纳分组、逻辑递进。思维基础架构就是先给出预期目标假设，以先验结论为引导，按特征工程统筹全局，逐步落实细节，先有结果，再有过程，最后再验证总结，并持续改进。
## 1.1. 目标
### 1.1.1. 结论
通常情况下，加油卡与车辆是一对一关系，其他多对多关系不是研究重点。
### 1.1.2. 特征工程与开发过程
（1）数据采集，构建分析样本原始数据

（2）特征工程与数据预处理，选择出待分析建模数据样本

（3）聚类，标注数据样本

（4）建模并训练模型，建立加油卡与号牌识别模型

（5）总结

## 1.2. 需求分析
我们在上班路上，经常能偶遇那个漂亮女孩，原因是她引起了我们的关注，其实，还有大叔、大姐等等。以此类推，类似同框出现的情况还有很多，加油站就上演了一幕。

加油站每天都有大量的车辆进出站，经营管理者很想知道客户用加油卡给那辆车加油，为客户建立画像，找到那个漂亮女孩。经营管理者希望通过大数据人工智能技术，识别加油卡与车辆号牌关系。

（1）在车辆进出站的时段内，出现过的加油卡，都有可能或者没有给车加油；

（2）一张加油卡可能给多辆车加油，一辆车可能使用多张卡加油；

（3）客户可能拥有多张加油卡、多辆车。

针对加油系统和号牌识别系统，经过几个月数据采集，车辆与加油卡重复相遇，加油行为用时与车辆在加油停留时长将是识别分析的数据基础。

基于周期性波形分析特征提取方法和相遇频次的假设建模分析。

## 1.3. 数据仿真
数据仿真参考汽车轻量化在线分析报告等媒体数据，仿真一年的数据，规则如下：

（1）一般汽车设计，一箱油行驶里程为600公里左右（不低于500公里），平均行驶里程为15000Km，中位数为12600Km；

（2）油耗数据正态分布：random.normalvariate（mu=8，sigma=1），整备质量：random.normalvariate（mu=1500，sigma=10）；

![image](https://github.com/xiaoyw71/Feature-engineering-machine-learning/blob/main/images/1.png)

（3）加油行为分布：

![image](https://github.com/xiaoyw71/Feature-engineering-machine-learning/blob/main/images/20201228141509917.png)

（4）非油购物占比为30%，随机购物为5%，卡充值频次为10%；

（5）加油枪流速37±1L/min的流速；

（6）其他随机处理，各项数据一般设定随机0.8到1.2，以及号牌不识别等随机事件。

# 2. 数据集
数据集来自加油卡使用记录和车牌识别记录，为了简化仿真实践，限定规则如下：加油卡使用，加油在前，非油消费在后。
## 2.1. 原始数据采集
采集数据包括：加油卡加油记录、加油卡非油消费记录、车辆进出站号牌识别记录。

## 2.2. 组合数据
按规则交叉组合加油卡加油记录集A和车辆进出站号牌识别记录集B，合并后的数据集合为C：

$C=A \cup B$
其中：
$c_k = a_i \cup b_j$
规则：
$a_i$加油时间大于$b_j$的进站时间，小于$b_j$的出站时间。

## 2.3. 仿真数据优势
仿真数据优势就是指我们知道分析的结果，有两种方式构建仿真数据，一是从现有实际数据中，提取已经有加油卡和车牌号匹配的数据；二是我们自己模拟仿真造数据或按数据规则标注，可以不精准。

我们在此实践中，仿真给出Flag特征为标识，加油卡与车牌号组合为两种情况，正确为True，错误为False。

# 3. 特征工程与数据预处理
我主要使用Pandas进行特征工程与数据预处理。
## 3.1. 特征提取
### 3.1.1. 时间特征提取
（1）拆分时间数据为年、月、日、时、分、秒；
（2）按加油卡分组，加油时间排序，提取加油卡加油间隔时间，同理提取车辆进站间隔时间；
> 加油间隔时间（日）、进站间隔时间（日）数据是一致的，从特征选择角度，直接删除。

（3）每种组合条件下，进站与加油间隔时间（带小数的时）、加油与出站间隔时间（带小数的时）。

### 3.1.2. 周期频次特征提取
基于周期性分析假设，加油卡与车牌号交叉组合，每种情况为一条记录，这样，每种情况对应采集到周期内相遇频次（以下简称频次）。

### 3.1.3. 周期时间性质特征提取
基于实用和业务现实角度，待分析样本数据集前提条件是，加油卡与车牌号相遇频次大于等于2。
（1）均值
（2）标准差
（3）偏度
（4）峰度
（5）最大值、最小值
```python
class ExploreDataAnalysis(object):
    def __init__(self,datas):     
        # 略
    def  Features_extra(self,drop_cols_name=[],times=1):
        df = self.df.drop(drop_cols_name,axis=1)
        # 取在同日、同一加油站相遇的次数
        # groupby中的as_index=False，对于聚合输出，返回以组标签作为索引的对象。仅与DataFrame输入相关。as_index = False实际上是“SQL风格”的分组输出。
        df_feature = df.groupby(['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag'], as_index=False)['fuel_interva'].count()        
        df_feature = df_feature.rename(columns={'fuel_interva':'times'})
        # 取相遇次数>times的数据集
        df_feature = df_feature[df_feature['times']>times].reset_index(drop=True) 
        # 依据相遇次数，筛选数据集
        df = pd.merge(df,df_feature[['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag']],how='inner',on=['IC_ID', 'License_plate','IC_ID_R', 'License_plate_R', 'Flag'],right_index=True)
        # 取均值
        df_feature1 = df[['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag','fuel_time','IC_time','Shop_time','Entry_time','Dep_time','Entry_interva','Shop_interva','Dep_interva','combos','Plate_interva','Entry_pre','Dep_pre']].groupby(['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag'], as_index=False).mean()
        df_feature = pd.merge(left=df_feature, right=df_feature1,how="left",on=['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag'])   #左连接 
        # 取标准差
        df_feature1 = df[['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag','fuel_time','IC_time','Shop_time','Entry_time','Dep_time', 'Entry_interva', 'Shop_interva','Dep_interva','Plate_interva','Entry_pre','Dep_pre']].groupby(['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag'], as_index=False).std()
                                  
        #修改为标准差列名
        df_feature1 = df_feature1.rename(columns={'fuel_time':'fuel_time_std','IC_time':'IC_time_std','Entry_time':'Entry_time_std','Dep_time':'Dep_time_std','Entry_interva':'Entry_interva_std', 'Dep_interva':'Dep_interva_std', 'Plate_interva':'Plate_interva_std','Shop_time':'Shop_time_std','Entry_pre':'Entry_pre_std','Dep_pre':'Dep_pre_std','Shop_interva':'Shop_interva_std'})
        df_feature = pd.merge(left=df_feature, right=df_feature1,how="left",on=['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R', 'Flag'])   #左连接                  
        self.data_feature = df_feature
        df_feature = df_feature.drop(['IC_ID', 'License_plate', 'IC_ID_R', 'License_plate_R'],axis=1)
        self.df_feature = df_feature
        
        return df_feature
```
> 其中，偏度与峰度依据后续的特征分析，由于不明显而略去。使用详见[[1]](https://xiaoyw.blog.csdn.net/article/details/110953695)

## 3.2. 特征分析
通过重点分析特征与输出预期（组合真伪）的关系，及对输出预期影响程度，为模型选择有用的特征进行建模分析。
### 3.2.1. 特征分析对象
抽取特征列表：
* 组合出现的频次：times
* 进出站时段内出现加油卡次数：combos
* 加油时长：fuel_time、fuel_time_std、...
* 进站时间的均值、标准差、最大值、最小值、偏度、峰度：Entry_time、Entry_time_std、...
* 加油时间的均值、标准差、最大值、最小值、偏度、峰度：IC_time、IC_time_std、...
* 购物时间的均值、标准差、最大值、最小值、偏度、峰度：Shop_time、Shop_time_std、...
* 出站时间的均值、标准差、最大值、最小值、偏度、峰度：Dep_time、Dep_time_std、...
* 进站与加油时间间隔的均值、标准差、最大值、最小值、偏度、峰度：Entry_interva、Entry_interva_std、...
* 加油与购物时间间隔的均值、标准差、最大值、最小值、偏度、峰度：Shop_interva、Shop_interva_std、...
* 购物与出站时间间隔的均值、标准差、最大值、最小值、偏度、峰度：Dep_interva、Dep_interva_std、...
* 进出站时间间隔的均值、标准差、最大值、最小值、偏度、峰度：Plate_interva、Plate_interva_std、...
* 进站与加油时间间隔/进出站时间间隔的均值、标准差、最大值、最小值、偏度、峰度：Entry_pre、Entry_pre_std、...
* 出站与加油时间间隔/进出站时间间隔的均值、标准差、最大值、最小值、偏度、峰度：Dep_pre、Dep_pre_std、...

### 3.2.2. 相关分析
使用皮尔逊相关系数分析输出目标（Flag）与其他各个特征的相关系数。上图为加油卡与号牌相遇重复2次及2次以上情况，下图为加油卡与号牌相遇重复4次及4次以上情况。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228153937819.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3hpYW95dw==,size_16,color_FFFFFF,t_70)
随着限定条件（提取数据）相遇频次增加，构建分析所选择的数据集，分析输出目标与频次（times）特征相关度随着频次增加而减弱，而分析输出目标与各项时长的标准差相关度增强，也体现出随着采集周期增加，周期性波动特征对模型影响将增强。

由于车辆加油周期性的原因，经常性随机相遇的概率是很高的，所以加油行为分析是提高准确度分析特征工程的发现，也有很合理的解释性。

其中，关于皮尔逊相关系数分析及绘图技术详见[[1]](https://xiaoyw.blog.csdn.net/article/details/110953695)。

### 3.2.3. 特征重要性分析
特征重要性对模型影响至关重要，如果只有少数几个特征有较高的重要分值，很可能对模型不利，应该再衍生特征来提高维度。使用随机森林进行特征重要性评估的思想，分析特征重要性及其排序，关于随机森林具体技术详见[[1]](https://xiaoyw.blog.csdn.net/article/details/110953695)。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228160228584.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3hpYW95dw==,size_16,color_FFFFFF,t_70)
初期分析时，频次“times”特征，占据到0.5分值，形成一枝独秀的效果，结果就成了“单因素”分析，准确度就停滞无法提高。然后，我挖掘出加油到出站间隔时长的标准差，使模型有了较大的改善。
```python
    def important_feature(self,cols_name):  
        # 模型输出目标为Flag
        y_df = df[['Flag']]
        X_df = df.drop('Flag',axis=1)
        x_train,x_text, y_train, y_test = train_test_split(X_df,y_df,test_size=0.1)
    
        forest_model = RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=-1)
        forest_model.fit(x_train, y_train)
        feat_labels = X_df.columns
        # feature_importances_  可以调取关于特征重要程度
        importances = forest_model.feature_importances_
        print("重要性：", importances)
        x_columns = X_df.columns
        indices = np.argsort(importances)[::-1]
        x_columns_indices = []
        for f in range(x_train.shape[1]):
            print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
            x_columns_indices.append(feat_labels[indices[f]])
         
        print(x_columns_indices)

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
```
## 3.3. 数据分布情况分析
我们知道在机器学习的世界中，以概率分布为核心的研究大都聚焦于正态分布。数据分布辅助我们选择算法模型。

我们可以从下面数据分布图，可以看出多数特征符合正态分布或者偏正态分布，对于选择聚类模型，可以倾向混合高斯模型（GaussianMixture）。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201229145116745.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3hpYW95dw==,size_16,color_FFFFFF,t_70)
实现上图代码如下：
```python
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
```
# 4. 聚类与数据标注
很多情况下，我们分析的数据样本，往往原始数据没有做好分类，需要我们标注数据的分类，在本案例里就是标注加油卡与车牌号组合是真是假。易操作可行的解决方案，通常采用聚类算法，通过聚类算法，标注数据集，给出加油卡与车牌号码组合为真或是假。

下面代码使用多种聚类算法聚类分析，对比分析聚类效果，选择适合的算法和结果。
```python
    def cluster_analysis(self):  
        Flag=list(df['Flag'])
        df=df.drop('Flag',axis=1)
        samples=df.values
        
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

        print('GaussianMixture') 
        gmm = GaussianMixture(n_components=2)
        gmm.fit(samples)
        labels = gmm.predict(samples)     
        
        df_cluster=pd.DataFrame({'labels':labels,'Flag':Flag})
        ct=pd.crosstab(df_cluster['labels'],df_cluster['Flag'])
        print(ct)

        return
```
使用K-Means、MiniBatchKMeans、Birch、GaussianMixture等四种算法进行聚类标注数据，聚类结果如下所示。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228162217660.png)
> 聚类结果中给出的标签labels，可以互换，只要分出不同的类别就可以，这里的0、1也是随机的，不信，可以试试。

在多次及不同数据集的情况下，混合高斯模型（GaussianMixture）都有较好的表现，分析原因有可能是我仿真数据过程中，使用较多的随机高斯数据分布。

> 曾使用多项式PolynomialFeatures进行升维，没有效果而放弃升维。

# 5. 加油卡与号牌识别模型
按特征工程过程，分析开发工作做的深入、扎实，则后续的机器学习、深度学习模型设计、训练与预测等过程将会非常顺利。
> 数据质量贯彻全过程始终，是课题研发成败重要因素。

## 5.1. 识别模型的输入与输出
### 5.1.1. 输入
（1）输入特征组成：['Flag', 'times','fuel_time','IC_time','Entry_time','Dep_time', 'Entry_interva', 'Dep_interva','combos',  'fuel_time_std','IC_time_std','Entry_time_std','Dep_time_std', 'Entry_interva_std','Dep_interva_std','Plate_interva','Entry_pre','Dep_pre','Plate_interva_std','Entry_pre_std','Dep_pre_std','Shop_time',  'Shop_interva','Shop_interva_std']  

（2）样本数及构成，总样本数：3525，其中真值为：1099，假值为：2426，真假不平衡（未做平衡处理）

> 做到平衡方式，一是删除一部分假值，或者，补充部分真值。

### 5.1.2. 输出
输出二分类预测模型，模型的输入参数为23个特征一组数据，输出为真伪（1/0）。

## 5.2. 集成机器学习算法XGBoost
对于中小型结构/表格数据时，现在一般认为基于决策树的算法是最佳方法。而基于决策树算法中的XGBoost，号称“比赛夺冠的必备大杀器”，横扫机器学习Kaggle、天池、DataCastle、Kesci等国内外数据竞赛罕逢敌手。

我选用binary:logistic实现二分类算法，识别加油卡与车牌号关系真假。通过训练模型，反馈给出在此模型下特征的重要程度分值。解读下图支撑模型解释性，频次和加油行为特征的标准差相对更重要些。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228162644522.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3hpYW95dw==,size_16,color_FFFFFF,t_70)
正如先验，XGBoost适合中小型结构/表格数据，是基于决策树的算法最佳方法。训练过程反馈给出8轮训练就得到最好的结论。
```python
    def train_Model_fit(self,num_class,normalization=True):
        params ={'learning_rate': 0.05,
          'max_depth': 5,                # 构建树的深度，越大越容易过拟合
          'num_boost_round':100,
          'objective': 'binary:logistic', # 二分类：binary:logistic
          'random_state': 7,
          'silent':0,
          'subsample':0.9,
          'min_child_weight':5,
          'eval_metric':['logloss','error','auc'], # 二分类情况
          'eta':0.3
        }
        dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
        dtest = xgb.DMatrix(self.x_text,label=self.y_test)
        
        res = xgb.cv(params,dtrain) 
        best_nround = res.shape[0] - 1
        watchlist = [(dtrain,'train'),(dtest,'eval')]
        evals_result = {}   
        model = xgb.train(params,dtrain,num_boost_round=best_nround,evals = watchlist,evals_result=evals_result)
        y_pred=model.predict(xgb.DMatrix(self.x_test))        
        
        model.save_model('XGboostClass.model')
        predictions = [round(value) for value in y_pred]
        # 计算准确率
        accuracy = accuracy_score(self.y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        # 显示重要特征
        plot_importance(model)
        plt.show()
```
我们通过“watchlist ”监控训练过程，效果如下图所示。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228162658936.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3hpYW95dw==,size_16,color_FFFFFF,t_70)
结果：Accuracy: 99.72%

## 5.2. Tensorflow BP神经网络
### 5.2.1. 神经网络结构设计
BP神经网络输入层为23个单元，接着网络各层分别为16、12、2，最后输出是2分类。关于Tensorflow神经网络各层的详解参见[[2]](https://xiaoyw.blog.csdn.net/article/details/108555559)。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228162859606.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3hpYW95dw==,size_16,color_FFFFFF,t_70)
其中：
* 神经元的激活函数采用relu
* 优化器使用AdagradOptimizer
* 损失函数采用softmax

### 5.2.2. 网络结构代码及训练情况
（1）构建BP神经网络模型及训练代码如下
```python
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
            h1_drop = tf.nn.dropout(h1,rate = 1 - keep_prob)   
            w2,b2 = NN(layers[0],layers[1],'2')   #定义第二层参数
            # 定义第二层隐藏层
            h2 = tf.nn.relu(tf.add(tf.matmul(h1_drop,w2),b2))
            w3,b3 = NN(layers[1],layers[2],'3')   #定义第三层参数
            y_conv = tf.nn.softmax(tf.add(tf.matmul(h2,w3),b3),name='y_conv')         
            #定义损失函数
            loss_func = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
            #学习率
            #learning_rate = #0.01
            lr = tf.Variable(learning_rate,dtype=tf.float32)
            train_step = tf.train.AdagradOptimizer(lr).minimize(loss_func)
            
            correct_pred = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            
            return x,y_,loss_func,train_step,correct_pred,keep_prob,accuracy,y_conv        

        batch_size = 20
        
        # 使用from_tensor_slices将数据放入队列，使用batch和repeat划分数据批次，且让数据序列无限延续
        dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        dataset = dataset.batch(batch_size).repeat()        
        iterator = dataset.make_one_shot_iterator()
        next_iterator = iterator.get_next()
        
        #定义神经网络的参数
        in_units = 23  #输入23个特征，返回一个正确与错误            
        x,y_,loss_,train_step,correct_pred,keep_prob,accuracy,y_conv = BP_NN(in_units,layers=[16,12,2], learning_rate=0.01) 
        log_loss= []
        log_acc = []
        #随机梯度下降算法训练参数
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())        
            for i in range(20000):
                batch_x,batch_y = sess.run(next_iterator)        
                _,loss,acc = sess.run([train_step,loss_,accuracy], feed_dict={x:batch_x,y_:batch_y,keep_prob:0.8})                
                if i%100 == 0:
                    print ("step： {}, total_loss： {},accuracy： {}".format(i, loss,acc))           # 用于趋势回归，预测值
                    log_loss.append(loss)
                    log_acc.append(acc)                        
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
                     
            print('测试集准确率：{}'.format(sess.run(tf.reduce_mean(tf.cast(test_correct, tf.float32)))))
            print('训练集准确率：{}'.format(sess.run(tf.reduce_mean(tf.cast(train_correct, tf.float32)))))

        return
```
（2）训练参数与过程

BP神经网络模型，训练过程如下图所示，其中，训练20000次，每次batch=20，学习率为0.01，而且使用Sklearn的MinMaxScaler对数据做归一化处理。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228164416666.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3hpYW95dw==,size_16,color_FFFFFF,t_70)
由于数据集样本有限，还存在数据不平衡的情况，模型及训练参数仅供参考，也只能说仿真数据集可能规律性更强些。
最后模型的准确率还是挺高的，希望能为实际生产提供指导意义。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228164437836.png)
全过程代码参见GitHub：[https://github.com/xiaoyw71/Feature-engineering-machine-learning](https://github.com/xiaoyw71/Feature-engineering-machine-learning)

# 6. 总结
通过训练集和测试集的验证，课题所提出识别加油卡与车号牌关系假设结论成立，数据分析过程可深入解释，支持加油卡与车号牌关系识别方法。

早先企业信息化信息化建设主要是为了加强规范化管理，服务于业务场景，与互联网行业先天大数据优势差距较大。现实的工业企业生产业务场景中，由于业务流程的特性，信息孤岛仍较为严重，很多也没有达到数字化，大多数情况下信息化数据及其特征还是偏少，样本也不足。

为此，我们在做大数据人工智能研究中，更多的仍需要依赖特征工程，应验了在业界广泛流传一句话：对于一个机器学习问题，数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。

对于本文中案例，先前研究更多的是依赖“频次”特征，而以频次为主的特征分析，聚类分析准确率只在70~85%间徘徊。而按周期波形分析方法提取特征，“频次”特征反而弱化了，加油行为则成为主要特征，这样更符合实际业务。

从训练模型角度分析，集成模型XGBoost在此场景下，有更好的适应性，也就是更能为实际工作服务。

如果后续有时间将更加深入分析，使用深度学习技术简化、弱化特征工程，使我们的工作有更多的灵活性选择，由于作者水平有限，欢迎交流讨论。

>文中的代码是以类的方法编写，只是截取主要核心片段，如需要请留言联系。

参考：

[[1]《大数据人工智能常用特征工程与数据预处理Python实践（2）》](https://xiaoyw.blog.csdn.net/article/details/110953695) CSDN博客 ，肖永威 ，2020年12月
[[2].《Tensorflow BP神经网络多输出模型在生产管理中应用实践》](https://xiaoyw.blog.csdn.net/article/details/108555559) CSDN博客 ， 肖永威 ，2020年9月
[[3].《XGBoost线性回归工控数据分析实践案例（原生篇）》](https://xiaoyw.blog.csdn.net/article/details/107900651) CSDB博客 ， 肖永威 ，2020年8月
[[4].《浅谈项目管理结构化思维》](https://xiaoyw.blog.csdn.net/article/details/105394163) CSDN博客 ， 肖永威 ，2020年4月
