#%% 
import numpy as np 
import keras as k 
from rules import vec2str, CARD_DIM

to_categorical = k.utils.to_categorical

NCARDGROUPS = 3  ##神经网络输入牌组数量
NCHANNEL = 9

##使用神经网络出牌的作弊机器人，它知道其他玩家手上的牌
## BOT类是演示用的，供ARENA类调用
## ARENA类要求所有BOT有如下4个成员函数
class BOT:
    def __init__(self, model, verbos=0, temperature=0.0):
        self.model = model
        self.verbos = verbos
        self.temperature = temperature

    ## 创建神经网络
    @classmethod
    def createmodel(cls):
        inputs = k.layers.Input(shape=(NCARDGROUPS, 18, NCHANNEL,))

        x = k.layers.Conv2D(256, (1, 3))(inputs)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)
        x = k.layers.Dropout(0.2)(x)

        for _ in range(6):
            x = k.layers.Conv2D(256, (1, 3))(x)
            x = k.layers.BatchNormalization()(x)
            x = k.layers.ReLU()(x)
            x = k.layers.Dropout(0.2)(x)

        x = k.layers.Conv2D(256, (NCARDGROUPS, 1))(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)
        x = k.layers.Dropout(0.2)(x)

        x = k.layers.Flatten()(x)
        x = k.layers.Dense(256)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)
        x = k.layers.Dropout(0.2)(x)
        
        outputs = k.layers.Dense(1, activation="sigmoid")(x)
        model = k.models.Model(inputs, outputs)
        model.compile(loss="binary_crossentropy",
                    optimizer="adam")
        return model
    
    @classmethod
    def initmodel(cls, path):
        model = cls.createmodel()
        model.save(path)

    def initiate(self, arena, role):
        self.arena = arena

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
        choices: 列表，每一项是长度15的数组
            所有合法出牌
        
        返回值：长度15的数组
            出这一手的局面估值最高
        '''
        
        ## 调用get_dizhu_win_prob计算choices里每种出牌后的局面估值
        scores = self.get_dizhu_win_prob(choices)
        if self.arena.pos != 0:
            scores = 1 - scores
        if self.temperature == 0:
            idx = np.argmax(scores)
        else:
            probs = scores ** (1/self.temperature)
            probs = probs/sum(probs)
            idx = np.random.choice(len(probs), p=probs)
        if self.verbos & 1:
            self.showChoices(5)
        return choices[idx]

    def get_dizhu_win_prob(self, choices):
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
        return self.model(xs).numpy()[:,0]

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
            print(scores[i], vec2str(choices[i]))
            num += 1
            if NUM and num == NUM:
                break



# %%
