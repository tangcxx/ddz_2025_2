#%% 
import numpy as np 
import keras as k 
import rules

to_categorical = k.utils.to_categorical

NCARDGROUPS = 3  ##神经网络输入牌组数量
CARD_DIM = rules.CARD_DIM  ## 牌组长度，15
NCHANNEL = 15

num2onehot = {0: np.array([0, 0, 0, 0]),
              1: np.array([1, 0, 0, 0]),
              2: np.array([1, 1, 0, 0]),
              3: np.array([1, 1, 1, 0]),
              4: np.array([1, 1, 1, 1])}

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

    @classmethod
    def encoding(cls, res, remain_pos, lastplay_pos):
        for i in range(15):
            card_hand_num = remain_pos[i]
            card_lastplay_num = lastplay_pos[i]
            res[i, 0:4] = num2onehot[card_hand_num]   ## 0:4 手牌
            res[i, 4:8] = num2onehot[card_lastplay_num]   ## 4:8 最后一手出牌

            if card_hand_num > 0:
                res[i, 7 + card_hand_num] = 1  ## 8:11 标记单张, 对子, 三张

        if remain_pos[13] == 1 and remain_pos[14] == 1:  ## 第11层是炸弹, 如果有王炸, 把第13位(对应小王)设为1
            res[13, 11] = 1

        for i in range(8):
            if np.all(res[i:(i+5), 0]):
                res[i:(i+5), 12] = 1           ## 12 顺子
        for i in range(10):
            if np.all(res[i:(i+3),1]):
                res[i:(i+3), 13] = 1           ## 13 连对
        for i in range(11):
            if np.all(res[i:(i+2),2]):
                res[i:(i+2), 14] = 1           ## 14 飞机
        return res
    
    @classmethod
    def getdata(cls, arena, choices):
        xs = np.zeros((len(choices), NCARDGROUPS, CARD_DIM, NCHANNEL), int)
        
        x = np.zeros((NCARDGROUPS, CARD_DIM, NCHANNEL), np.int8)
        cls.encoding(x[arena.b1], arena.remain[arena.b1], arena.lastplay[arena.b1])
        cls.encoding(x[arena.b2], arena.remain[arena.b2], np.zeros(15, np.int8))
        xs = np.repeat(x[np.newaxis, :, :, :], len(choices), axis=0)
        for idx_choice, choice in enumerate(choices):
            remain_pos = arena.remain[arena.pos] - choice
            cls.encoding(xs[idx_choice, arena.pos], remain_pos, choice)

        return xs

    def netchoose(self, choices):
        # '''
        # choices: 列表, 每一项是长度15的数组
        #     所有合法出牌
        
        # 返回值: 长度15的数组
        #     出这一手的局面估值最高
        # '''
        # if np.random.random() < self.epsilon:
        #     idx = np.random.choice(len(choices))
        #     self.xs.append(self.getdata(self.arena, choices[idx:idx+1]))
        #     return choices[idx]

        ## 调用 get_dizhu_win_probs 计算choices里每种出牌后的局面估值
        xs = self.getdata(self.arena, choices)
        dizhu_win_probs = self.model(xs).numpy()[:,0]
        scores = dizhu_win_probs if self.arena.pos == 0 else 1 - dizhu_win_probs
        
        if self.verbos & 1:  ## 打印出牌选择及相应的估值
            for i in np.argsort(scores)[::-1][0:5]:
                print(scores[i], rules.vec2str(choices[i]))
        
        idx = np.argmax(scores)  ## 找到最大的那一项的序号（下标）

        self.xs.append(xs[idx:(idx+1)])  ## 记录状态
        return choices[idx]

# %%
