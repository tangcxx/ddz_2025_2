#%% 
import numpy as np 
import keras as k 
import rules

to_categorical = k.utils.to_categorical

NCARDGROUPS = 3  ##神经网络输入牌组数量
CARD_DIM = rules.CARD_DIM  ## 牌组长度，15
NCARDSPARAM = NCARDGROUPS * CARD_DIM  ##代表牌组的输入的长度
NCHANNEL = 9

##使用神经网络出牌的作弊机器人，它知道其他玩家手上的牌
## BOT类是演示用的，供ARENA类调用
## ARENA类要求所有BOT有如下4个成员函数
class BOT:
    def __init__(self, model, verbos=0, epsilon=0):
        self.model = model
        self.verbos = verbos
        self.epsilon = epsilon
        self.xs = []
        self.v_est_max = []


    ## 创建神经网络
    @classmethod
    def createmodel(cls):
        inputs = k.layers.Input(shape=(NCARDGROUPS, 18, NCHANNEL,))

        x = k.layers.Conv2D(512, (1, 3), strides=(1,2))(inputs)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)
        x = k.layers.Dropout(0.2)(x)

        x = k.layers.Conv2D(512, (1, 3), strides=(1,2))(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)
        x = k.layers.Dropout(0.2)(x)

        x = k.layers.Conv2D(512, (1, 3))(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)
        x = k.layers.Dropout(0.2)(x)

        x = k.layers.Conv2D(512, (1, 1))(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)
        x = k.layers.Dropout(0.2)(x)

        x = k.layers.Conv2D(512, (NCARDGROUPS, 1))(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)
        x = k.layers.Dropout(0.2)(x)

        x = k.layers.Flatten()(x)

        x = k.layers.Dense(256)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)
        x = k.layers.Dropout(0.2)(x)
        
        outputs = k.layers.Dense(1, activation="tanh")(x)
        model = k.models.Model(inputs, outputs)
        model.compile(loss="mean_squared_error",
                    optimizer="adam")
        return model
    
    @classmethod
    def initmodel(cls, path):
        model = cls.createmodel()
        model.save(path)

    def initiate(self, arena, role):
        self.arena = arena
        self.xs = []
        self.v_est_max = []

    def play(self):
        ## 获取所有合法出牌
        choices = self.arena.getChoices()
        ## 调用 netchoose 选择一手出牌
        return self.netchoose(choices)

    def update(self):
        pass

    @classmethod
    def getdata(cls, arena):
        data = np.zeros((NCARDGROUPS, CARD_DIM, NCHANNEL), int)  ##创建工为NCARDSPARAM的数组，保存另一些局面数据
        data2 = np.zeros((NCARDGROUPS, 18, NCHANNEL), int)
        
        for pos in range(3):
            for num in range(4):
                data[pos, :, num] = (arena.remain[pos] > num)
                if pos != arena.pos:
                    data[pos, :, num+4] = (arena.lastplay[pos] > num)

        data2[:,0:12,0:8] = data[:,0:12,0:8]
        data2[:,13,0:8] = data[:,12,0:8]
        data2[:,15,0:8] = data[:,13,0:8]
        data2[:,17,0:8] = data[:,14,0:8]

        data2[arena.pos, :, 8] = 1

        data2.shape = (1, NCARDGROUPS, 18, NCHANNEL)
        
        return data2

    def eval(self, arena=None):
        arena = arena or self.arena
        data = self.getdata(arena)
        return self.model(data)[0, 0].numpy()


    def netchoose(self, choices):
        '''
        choices: 列表, 每一项是长度15的数组
            所有合法出牌
        
        返回值: 长度15的数组
            出这一手的局面估值最高
        '''
        
        ## 调用 get_dizhu_scores 计算choices里每种出牌后的局面估值
        xs, dizhu_scores = self.get_dizhu_scores(choices)
        if self.arena.pos == 0:
            idx = np.argmax(dizhu_scores)  ## 找到最大的那一项的序号（下标）
        else:
            idx = np.argmin(dizhu_scores)

        self.v_est_max.append(dizhu_scores[idx])  ## 记录可能的状态评分，用于fit上一个状态

        if np.random.random() < self.epsilon:
            idx = np.random.choice(len(choices))

        self.xs.append(xs[idx:(idx+1)])  ## 记录状态

        if self.verbos & 1:
            self.showChoices(5)
        return choices[idx]

    def get_dizhu_scores(self, choices):
        '''
        choices: 列表, 每一项是长度15的数组
            所有合法出牌
        
        返回值: 和choices一样长的列表, 每一项是数字(浮点, 0到1之间)
            对应每一种出牌后的局面估值
        '''
        xs = []
        ##循环每一种出牌
        for choice in choices: 
            # print("choice ", vec2str(choice))
            cp = self.arena.copy()  ##复制当前状态，后续操作在副本上进行，避免破坏当前状态
            cp.update(choice)  ##出牌
            xs.append(self.getdata(cp))
        xs = np.concatenate(xs)
        return xs, self.model(xs).numpy()[:,0]

    ## 调试用，打印某个局面下所有的出牌选择及相应的估值
    def showChoices(self, NUM=None):
        arena = self.arena
        choices = arena.getChoices()
        scores = []
        for choice in choices:
            cp = arena.copy()
            cp.update(choice)
            score = self.eval(cp)
            scores.append(score)
        scores = scores if arena.pos == 0 else 1 - np.array(scores, 'float32')

        num = 0
        for i in np.argsort(scores)[::-1]:
            print(scores[i], rules.vec2str(choices[i]))
            num += 1
            if NUM and num == NUM:
                break



# %%
