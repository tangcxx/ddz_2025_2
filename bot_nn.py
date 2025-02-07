import numpy as np 
import keras as k 
import rules

to_categorical = k.utils.to_categorical

NCARDGROUPS = 6  ##神经网络输入牌组数量
CARD_DIM = rules.CARD_DIM  ## 牌组长度，15
NCARDSPARAM = NCARDGROUPS * CARD_DIM  ##代表牌组的输入的长度
NCARDBIT = 5  ## 每种牌（牌组的每一项）用几位表示
NCHANNEL = 8

##使用神经网络出牌的作弊机器人，它知道其他玩家手上的牌
## BOT类是演示用的，供ARENA类调用
## ARENA类要求所有BOT有如下4个成员函数
class BOT:
    def __init__(self, model, verbos=0, epsilon=0):
        self.model = model
        self.verbos = verbos
        self.epsilon = epsilon

    ## 创建神经网络
    @classmethod
    def createmodel(cls):
        inputs = k.layers.Input(shape=(NCARDGROUPS, CARD_DIM, NCHANNEL,))
        x = k.layers.Conv2D(128, (1, CARD_DIM), activation="relu")(inputs)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Conv2D(128, (NCARDGROUPS, 1), activation="relu")(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Flatten()(x)
        x = k.layers.Dense(128, activation="relu")(x)
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
        if np.random.random() < self.epsilon:
            idx = np.random.choice(len(choices))
            return choices[idx]
        ## 调用 netchoose 选择一手出牌
        return self.netchoose(choices)

    def update(self):
        pass

    @classmethod
    def getdata(cls, arena):
        data = np.zeros((NCARDGROUPS, CARD_DIM, NCHANNEL), int)  ##创建工为NCARDSPARAM的数组，保存另一些局面数据
        data[0, :, 0:5] = to_categorical(arena.remain[0], NCARDBIT)
        data[1, :, 0:5] = to_categorical(arena.remain[1], NCARDBIT)
        data[2, :, 0:5] = to_categorical(arena.remain[2], NCARDBIT)
        data[3, :, 0:5] = to_categorical(arena.lastplay[0], NCARDBIT)
        data[4, :, 0:5] = to_categorical(arena.lastplay[1], NCARDBIT)
        data[5, :, 0:5] = to_categorical(arena.lastplay[2], NCARDBIT)
        data[:, :, 5:8] = to_categorical(arena.pos, 3)
        data.shape = (1, NCARDGROUPS, CARD_DIM, NCHANNEL)
        return data

    def eval(self, arena=None):
        arena = arena or self.arena
        data = self.getdata(arena)
        # return self.model.predict(data, verbose = 0)[0, 0]
        return self.model(data)[0, 0].numpy()


    def netchoose(self, choices):
        '''
        choices: 列表，每一项是长度15的数组
            所有合法出牌
        
        返回值：长度15的数组
            出这一手的局面估值最高
        '''
        
        ## 调用netscores计算choices里每种出牌后的局面估值
        scores = self.netscores(choices)
        if self.arena.pos == 0:
            idx = np.argmax(scores)  ## 找到最大的那一项的序号（下标）
        else:
            idx = np.argmin(scores)
        if self.verbos & 1:
            self.showChoices(5)
        return choices[idx]

    def netscores(self, choices):
        '''
        choices: 列表，每一项是长度15的数组
            所有合法出牌
        
        返回值：和choices一样长的列表，每一项是数字（浮点，0到1之间）
            对应每一种出牌后的局面估值
        '''
        scores = []  ##保存返回值
        ##循环每一种出牌
        for choice in choices: 
            # print("choice ", rules.vec2str(choice))
            cp = self.arena.copy()  ##复制当前状态，后续操作在副本上进行，避免破坏当前状态
            cp.update(choice)  ##出牌
            score = self.eval(cp) ##估值
            scores.append(score) ##将估值添加到列表里
        return scores

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


