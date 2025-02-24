#%% 
import numpy as np 
import keras as k 
import rules

to_categorical = k.utils.to_categorical

NCARDGROUPS = 3  ##神经网络输入牌组数量
CARD_DIM = rules.CARD_DIM  ## 牌组长度, 15
NCHANNEL = 9

CARDS_SHAPE = (15, 8,)
COUNT_REMAINS_SHAPE = (3,)
HISTORY_SHAPE = (160, 60,)


##使用神经网络出牌的作弊机器人, 它知道其他玩家手上的牌
## BOT类是演示用的, 供ARENA类调用
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

        inputs_cards = k.layers.Input(shape=CARDS_SHAPE) # 手牌，余牌
        inputs_count_remains = k.layers.Input(shape=COUNT_REMAINS_SHAPE) # 手牌数量
        inputs_history = k.layers.Input(shape=HISTORY_SHAPE) # 历史记录

        x = k.layers.Flatten()(inputs_cards)

        _, history, _ = k.layers.LSTM(256, return_sequences=False, return_state=True)(inputs_history)
        history = k.layers.Dropout(0.2)(history)

        x = k.layers.concatenate([x, inputs_count_remains, history])

        x = k.layers.Dense(512)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)
        x = k.layers.Dropout(0.2)(x)

        x = k.layers.Dense(512)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)
        x = k.layers.Dropout(0.2)(x)

        x = k.layers.Dense(256)(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.ReLU()(x)
        x = k.layers.Dropout(0.2)(x)

        outputs = k.layers.Dense(1, activation="sigmoid")(x)
        model = k.models.Model([inputs_cards, inputs_count_remains, inputs_history], outputs)
        model.compile(loss="binary_crossentropy",
                    optimizer="adam")
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

    @classmethod
    def getdata(cls, arena):
        data_cards = np.zeros(CARDS_SHAPE, int)
        pos = arena.pos
        remain_others = arena.remain[(pos + 1) % 3] = arena.remain[(pos + 2) % 3]
        for i in range(4):
            data_cards[:, i] = arena.remain[pos] > i
            data_cards[:, i+4] = remain_others > i
        data_cards.shape = (1, *CARDS_SHAPE)

        data_count_remains = np.sum(arena.remain, axis=1)
        data_count_remains.shape = (1, *COUNT_REMAINS_SHAPE)
        
        data_history = np.zeros(HISTORY_SHAPE, int)
        for i, cards in enumerate(arena.history):
            data_history[i] = cards
        data_history.shape = (1, *HISTORY_SHAPE)

        return [data_cards, data_count_remains, data_history]

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
        choices: 列表, 每一项是长度15的数组
            所有合法出牌
        
        返回值: 和choices一样长的列表, 每一项是数字 (浮点, 0到1之间)
            对应每一种出牌后的局面估值
        '''
        xs = []
        ##循环每一种出牌
        for choice in choices: 
            # print("choice ", vec2str(choice))
            cp = self.arena.copy()  ##复制当前状态, 后续操作在副本上进行, 避免破坏当前状态
            cp.update(choice)  ##出牌
            xs.append(self.getdata(cp))
        xs = np.concatenate(xs)
        return xs, self.model(xs).numpy()[:,0]

    ## 调试用, 打印某个局面下所有的出牌选择及相应的估值
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
