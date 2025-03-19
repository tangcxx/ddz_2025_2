#%% 
import numpy as np 
import keras as k 
import rules

to_categorical = k.utils.to_categorical

NCARDGROUPS = 3  ##神经网络输入牌组数量
CARD_DIM = rules.CARD_DIM  ## 牌组长度，15
NCHANNEL = 16

##使用神经网络出牌的作弊机器人，它知道其他玩家手上的牌
## BOT类是演示用的，供ARENA类调用
## ARENA类要求所有BOT有如下4个成员函数
class BOT:
    def __init__(self, model, verbos=0, epsilon=0):
        self.model = model
        self.verbos = verbos
        self.epsilon = epsilon
        self.xs = []

    ## 创建神经网络
    @classmethod
    def createmodel(cls):
        inputs = k.layers.Input(shape=(NCARDGROUPS, CARD_DIM, NCHANNEL,))

        x = k.layers.Conv2D(512, (1, CARD_DIM))(inputs)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)
        x = k.layers.Dropout(0.2)(x)

        x = k.layers.Conv2D(512, (NCARDGROUPS, 1))(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)
        x = k.layers.Dropout(0.2)(x)

        x = k.layers.Flatten()(x)

        x = k.layers.Dense(512)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)
        x = k.layers.Dropout(0.2)(x)

        x = k.layers.Dense(512)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)
        x = k.layers.Dropout(0.2)(x)

        x = k.layers.Dense(512)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)
        x = k.layers.Dropout(0.2)(x)

        outputs = k.layers.Dense(1, activation="sigmoid")(x)
        model = k.models.Model(inputs, outputs)
        model.compile(loss="binary_crossentropy",
                    optimizer=k.optimizers.Adam(learning_rate=1e-4))
        return model
    
    @classmethod
    def initmodel(cls, path):
        model = cls.createmodel()
        model.save(path)

    def initiate(self, arena, role):
        self.arena = arena
        self.xs = []

    def play(self):
        ## 获取所有合法出牌
        choices = self.arena.getChoices()
        ## 调用 netchoose 选择一手出牌
        return self.netchoose(choices)

    def update(self):
        pass

    # 0,1,2,3层: 手牌
    # 4,5,6,7: 上一轮出牌
    # 8: 当前出牌人
    # 9: 单张
    # 10: 对子
    # 11: 三张
    # 12：炸弹
    # 13：顺子
    # 14: 连对
    # 15: 飞机
    @classmethod
    def getdata(cls, arena):
        data = np.zeros((NCARDGROUPS, CARD_DIM, NCHANNEL), int)
        remain = arena.remain
        lastplay = arena.lastplay
        
        data[:, :, 0] = remain > 0                      ## 0
        data[:, :, 1] = remain > 1                      ## 1
        data[:, :, 2] = remain > 2                     ## 2
        data[:, :, 3] = remain > 3                     ## 3
        
        data[:, :, 4] = lastplay > 0                     ## 4
        data[:, :, 5] = lastplay > 1                     ## 5
        data[:, :, 6] = lastplay > 2                     ## 6
        data[:, :, 7] = lastplay > 3                     ## 7
        
        data[arena.pos, :, 4:8] = 0                        ## 出牌人的上一轮出牌不考虑

        data[arena.pos, :, 8] = 1                        ## 8
        
        data[:, :, 9] = remain == 1                      ## 9
        data[:, :, 10] = remain == 2                     ## 10
        data[:, :, 11] = remain == 3                     ## 11
        data[:, :, 12] = remain == 4                     ## 12
        for pos in range(3):
            if remain[pos, 13] == 1 and remain[pos, 14] == 1:
                data[pos, 13, 12] = 1

        for pos in range(3):
            for i in range(8):
                if np.all(remain[pos, i:(i+5)] >= 1):
                    data[pos, i:(i+5), 13] = 1           ## 13
            for i in range(10):
                if np.all(remain[pos, i:(i+3)] >= 2):
                    data[pos, i:(i+3), 14] = 1           ## 14  
            for i in range(11):
                if np.all(remain[pos, i:(i+2)] >= 3):
                    data[pos, i:(i+2), 15] = 1           ## 15

        data.shape = (1, NCARDGROUPS, CARD_DIM, NCHANNEL)
        return data

    def eval(self, arena=None):
        arena = arena or self.arena
        data = self.getdata(arena)
        return self.model(data)[0, 0].numpy()


    def netchoose(self, choices):
        '''
        choices: 列表，每一项是长度15的数组
            所有合法出牌
        
        返回值：长度15的数组
            出这一手的局面估值最高
        '''
        if np.random.random() < self.epsilon:
            idx = np.random.choice(len(choices))
            cp = self.arena.copy()
            cp.update(choices[idx])
            self.xs.append(self.getdata(cp))
            return choices[idx]

        ## 调用 get_dizhu_win_probs 计算choices里每种出牌后的局面估值
        xs, dizhu_win_probs = self.get_dizhu_win_probs(choices)
        if self.arena.pos == 0:
            idx = np.argmax(dizhu_win_probs)  ## 找到最大的那一项的序号（下标）
        else:
            idx = np.argmin(dizhu_win_probs)
        if self.verbos & 1:
            self.showChoices(5)

        self.xs.append(xs[idx:(idx+1)])  ## 记录状态
        return choices[idx]

    def get_dizhu_win_probs(self, choices):
        '''
        choices: 列表，每一项是长度15的数组
            所有合法出牌
        
        返回值：和choices一样长的列表，每一项是数字（浮点，0到1之间）
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
        xs = []
        for choice in choices:
            cp = arena.copy()
            cp.update(choice)
            xs.append(self.getdata(cp))
        xs = np.concatenate(xs)
        scores = self.model(xs).numpy()[:,0]
        if arena.pos != 0:
            scores = 1 - scores

        num = 0
        for i in np.argsort(scores)[::-1]:
            print(scores[i], rules.vec2str(choices[i]))
            num += 1
            if NUM and num == NUM:
                break



# %%
