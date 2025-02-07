# 策略模型
# 作弊
# 三个模型：出牌和带牌（actor)，局面评估(critic)
# model_policy, model_discard, model_value

#流程： 
# 发牌
# 依次出牌
# # 当前状态（手牌，上一轮出牌，当前出牌人位置）
# # 输出策略和状态评分
# # 本次出牌的奖励，按最终胜负，如赢： 1 - 评分，如输：-1 - 评分，更新出牌网络
# # 评分，如赢：1，如输：-1，更新评分网络
#%%
import numpy as np

#%%
import rules
from rules import list2str, list2vec, vec2list, vec2str, str2vec


# 编码动作

#%%
ACTION_LISTS = [
    [],
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10],
    [11],
    [12],
    [13],
    [14],
    [0,0],
    [1,1],
    [2,2],
    [3,3],
    [4,4],
    [5,5],
    [6,6],
    [7,7],
    [8,8],
    [9,9],
    [10,10],
    [11,11],
    [12,12],
    [0,0,0],
    [1,1,1],
    [2,2,2],
    [3,3,3],
    [4,4,4],
    [5,5,5],
    [6,6,6],
    [7,7,7],
    [8,8,8],
    [9,9,9],
    [10,10,10],
    [11,11,11],
    [12,12,12],
    [0,1,2,3,4],
    [1,2,3,4,5],
    [2,3,4,5,6],
    [3,4,5,6,7],
    [4,5,6,7,8],
    [5,6,7,8,9],
    [6,7,8,9,10],
    [7,8,9,10,11],
    [0,1,2,3,4,5],
    [1,2,3,4,5,6],
    [2,3,4,5,6,7],
    [3,4,5,6,7,8],
    [4,5,6,7,8,9],
    [5,6,7,8,9,10],
    [6,7,8,9,10,11],
    [0,1,2,3,4,5,6],
    [1,2,3,4,5,6,7],
    [2,3,4,5,6,7,8],
    [3,4,5,6,7,8,9],
    [4,5,6,7,8,9,10],
    [5,6,7,8,9,10,11],
    [0,1,2,3,4,5,6,7],
    [1,2,3,4,5,6,7,8],
    [2,3,4,5,6,7,8,9],
    [3,4,5,6,7,8,9,10],
    [4,5,6,7,8,9,10,11],
    [0,1,2,3,4,5,6,7,8],
    [1,2,3,4,5,6,7,8,9],
    [2,3,4,5,6,7,8,9,10],
    [3,4,5,6,7,8,9,10,11],
    [0,1,2,3,4,5,6,7,8,9],
    [1,2,3,4,5,6,7,8,9,10],
    [2,3,4,5,6,7,8,9,10,11],
    [0,1,2,3,4,5,6,7,8,9,10],
    [1,2,3,4,5,6,7,8,9,10,11],
    [0,1,2,3,4,5,6,7,8,9,10,11],
    [0,0,1,1,2,2],
    [1,1,2,2,3,3],
    [2,2,3,3,4,4],
    [3,3,4,4,5,5],
    [4,4,5,5,6,6],
    [5,5,6,6,7,7],
    [6,6,7,7,8,8],
    [7,7,8,8,9,9],
    [8,8,9,9,10,10],
    [9,9,10,10,11,11],
    [0,0,1,1,2,2,3,3],
    [1,1,2,2,3,3,4,4],
    [2,2,3,3,4,4,5,5],
    [3,3,4,4,5,5,6,6],
    [4,4,5,5,6,6,7,7],
    [5,5,6,6,7,7,8,8],
    [6,6,7,7,8,8,9,9],
    [7,7,8,8,9,9,10,10],
    [8,8,9,9,10,10,11,11],
    [0,0,1,1,2,2,3,3,4,4],
    [1,1,2,2,3,3,4,4,5,5],
    [2,2,3,3,4,4,5,5,6,6],
    [3,3,4,4,5,5,6,6,7,7],
    [4,4,5,5,6,6,7,7,8,8],
    [5,5,6,6,7,7,8,8,9,9],
    [6,6,7,7,8,8,9,9,10,10],
    [7,7,8,8,9,9,10,10,11,11],
    [0,0,1,1,2,2,3,3,4,4,5,5],
    [1,1,2,2,3,3,4,4,5,5,6,6],
    [2,2,3,3,4,4,5,5,6,6,7,7],
    [3,3,4,4,5,5,6,6,7,7,8,8],
    [4,4,5,5,6,6,7,7,8,8,9,9],
    [5,5,6,6,7,7,8,8,9,9,10,10],
    [6,6,7,7,8,8,9,9,10,10,11,11],
    [0,0,1,1,2,2,3,3,4,4,5,5,6,6],
    [1,1,2,2,3,3,4,4,5,5,6,6,7,7],
    [2,2,3,3,4,4,5,5,6,6,7,7,8,8],
    [3,3,4,4,5,5,6,6,7,7,8,8,9,9],
    [4,4,5,5,6,6,7,7,8,8,9,9,10,10],
    [5,5,6,6,7,7,8,8,9,9,10,10,11,11],
    [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7],
    [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8],
    [2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9],
    [3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10],
    [4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11],
    [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8],
    [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9],
    [2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10],
    [3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11],
    [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9],
    [1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10],
    [2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11],
    [0,0,0,1,1,1],
    [1,1,1,2,2,2],
    [2,2,2,3,3,3],
    [3,3,3,4,4,4],
    [4,4,4,5,5,5],
    [5,5,5,6,6,6],
    [6,6,6,7,7,7],
    [7,7,7,8,8,8],
    [8,8,8,9,9,9],
    [9,9,9,10,10,10],
    [10,10,10,11,11,11],
    [0,0,0,1,1,1,2,2,2],
    [1,1,1,2,2,2,3,3,3],
    [2,2,2,3,3,3,4,4,4],
    [3,3,3,4,4,4,5,5,5],
    [4,4,4,5,5,5,6,6,6],
    [5,5,5,6,6,6,7,7,7],
    [6,6,6,7,7,7,8,8,8],
    [7,7,7,8,8,8,9,9,9],
    [8,8,8,9,9,9,10,10,10],
    [9,9,9,10,10,10,11,11,11],
    [0,0,0,1,1,1,2,2,2,3,3,3],
    [1,1,1,2,2,2,3,3,3,4,4,4],
    [2,2,2,3,3,3,4,4,4,5,5,5],
    [3,3,3,4,4,4,5,5,5,6,6,6],
    [4,4,4,5,5,5,6,6,6,7,7,7],
    [5,5,5,6,6,6,7,7,7,8,8,8],
    [6,6,6,7,7,7,8,8,8,9,9,9],
    [7,7,7,8,8,8,9,9,9,10,10,10],
    [8,8,8,9,9,9,10,10,10,11,11,11],
    [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4],
    [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5],
    [2,2,2,3,3,3,4,4,4,5,5,5,6,6,6],
    [3,3,3,4,4,4,5,5,5,6,6,6,7,7,7],
    [4,4,4,5,5,5,6,6,6,7,7,7,8,8,8],
    [5,5,5,6,6,6,7,7,7,8,8,8,9,9,9],
    [6,6,6,7,7,7,8,8,8,9,9,9,10,10,10],
    [7,7,7,8,8,8,9,9,9,10,10,10,11,11,11],
    [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5],
    [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6],
    [2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7],
    [3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8],
    [4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9],
    [5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10],
    [6,6,6,7,7,7,8,8,8,9,9,9,10,10,10,11,11,11],
    [0,0,0,0],
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3],
    [4,4,4,4],
    [5,5,5,5],
    [6,6,6,6],
    [7,7,7,7],
    [8,8,8,8],
    [9,9,9,9],
    [10,10,10,10],
    [11,11,11,11],
    [12,12,12,12],
    [13,14]
]
# %%
ACTIONS = [list2vec(l) for l in ACTION_LISTS]
# %%

WITH_SIDES = [0, 1]

SIDE_SINGLE_LISTS = [
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
]

SIDE_PAIR_LISTS = [
    0,1,2,3,4,5,6,7,8,9,10,11,12
]

# %%
import tensorflow as tf
from tensorflow.keras import layers, models

#%%
def create_model():
    input_shape = (6, 15, 8)  # 输入形状

    # 输入层
    inputs = layers.Input(shape=input_shape)

    # 卷积层：1x3 卷积
    x = layers.Conv2D(filters=256, kernel_size=(1, 3), padding='same')(inputs)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # 多次重复卷积层
    for _ in range(6):  # 假设需要重复6次
        x = layers.Conv2D(filters=256, kernel_size=(1, 3), padding='same')(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

    # 最后一个卷积层：6x1 卷积
    x = layers.Conv2D(filters=512, kernel_size=(6, 1), padding='valid')(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    # Flatten层
    x = layers.Flatten()(x)

    # 输出层
    # output01 = layers.Dense(128)(x)  # 输出1个数
    # output01 = layers.BatchNormalization()(output01)  # 输出1个数
    # output01 = layers.ReLU()(output01)  # 输出1个数
    # output01 = layers.Dropout(0.2)(output01)  # 输出1个数
    output01 = layers.Dense(1, activation='tanh', name='output01')(x)  # 输出1个数
    
    # output02 = layers.Dense(256)(x)  # 输出1个数
    # output02 = layers.BatchNormalization()(output02)  # 输出1个数
    # output02 = layers.ReLU()(output02)  # 输出1个数
    # output02 = layers.Dropout(0.2)(output02)  # 输出1个数
    output02 = layers.Dense(189, activation='softmax', name='output02')(x)  # 输出189个数
    
    output03 = layers.Dense(2, activation='softmax', name='output03')(x)  # 输出2个数
    output04 = layers.Dense(15, activation='softmax', name='output04')(x)  # 输出15个数
    output05 = layers.Dense(15, activation='softmax', name='output05')(x)  # 输出15个数
    output06 = layers.Dense(15, activation='softmax', name='output06')(x)  # 输出15个数
    output07 = layers.Dense(15, activation='softmax', name='output07')(x)  # 输出15个数
    output08 = layers.Dense(15, activation='softmax', name='output08')(x)  # 输出15个数
    output09 = layers.Dense(13, activation='softmax', name='output09')(x)  # 输出13个数
    output10 = layers.Dense(13, activation='softmax', name='output10')(x)  # 输出13个数
    output11 = layers.Dense(13, activation='softmax', name='output11')(x)  # 输出13个数
    output12 = layers.Dense(13, activation='softmax', name='output12')(x)  # 输出13个数

    # 创建模型
    model = models.Model(inputs=inputs, outputs=[
        output01,
        output02,
        output03,
        output04,
        output05,
        output06,
        output07,
        output08,
        output09,
        output10,
        output11,
        output12
    ])

    return model

# 创建模型
model = create_model()

# 编译模型
# model.compile(optimizer='adam', 
#               loss={
#                   'output01': 'mean_squared_error',
#                   'output02': 'categorical_crossentropy',
#                   'output03': 'categorical_crossentropy',
#                   'output04': 'categorical_crossentropy',
#                   'output05': 'categorical_crossentropy',
#                   'output06': 'categorical_crossentropy',
#                   'output07': 'categorical_crossentropy',
#                   'output08': 'categorical_crossentropy',
#                   'output09': 'categorical_crossentropy',
#                   'output10': 'categorical_crossentropy',
#                   'output11': 'categorical_crossentropy',
#                   'output12': 'categorical_crossentropy',
#               })

# 打印模型摘要
model.summary()
# %%
smps = np.random.normal(loc=0.0, scale=1.0, size=(10, 6, 15, 8))
# %%
y_pred = model(smps)
# %%
y_pred[2]
# %%
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 

def updatemodel(model, optimizer, xs, ys):
    with tf.GradientTape() as tape:
        ys_pred = model(xs)

        actions = np.array([np.random.choice(8, p=prob) for prob in probs.numpy()])
        rewards = np.array([run(state, action) for state, action in zip(xs, actions)])
        rewards = rewards - optimal_values
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

        action_probs = tf.gather_nd(probs, [[i, a] for i, a in enumerate(actions)])
        log_probs = tf.math.log(action_probs)
        loss = -tf.reduce_mean(log_probs * rewards)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    
#%%
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
    def create_model(cls):
        input_shape = (6, 15, 8)  # 输入形状

        # 输入层
        inputs = layers.Input(shape=input_shape)

        # 卷积层：1x3 卷积
        x = layers.Conv2D(filters=256, kernel_size=(1, 3), padding='same')(inputs)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        # 多次重复卷积层
        for _ in range(6):  # 假设需要重复6次
            x = layers.Conv2D(filters=256, kernel_size=(1, 3), padding='same')(x)
            x = layers.ReLU()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        # 最后一个卷积层：6x1 卷积
        x = layers.Conv2D(filters=512, kernel_size=(6, 1), padding='valid')(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)

        # Flatten层
        x = layers.Flatten()(x)

        # 输出层
        # output01 = layers.Dense(128)(x)  # 输出1个数
        # output01 = layers.BatchNormalization()(output01)  # 输出1个数
        # output01 = layers.ReLU()(output01)  # 输出1个数
        # output01 = layers.Dropout(0.2)(output01)  # 输出1个数
        output01 = layers.Dense(1, activation='tanh', name='output01')(x)  # 输出1个数
        
        # output02 = layers.Dense(256)(x)  # 输出1个数
        # output02 = layers.BatchNormalization()(output02)  # 输出1个数
        # output02 = layers.ReLU()(output02)  # 输出1个数
        # output02 = layers.Dropout(0.2)(output02)  # 输出1个数
        output02 = layers.Dense(189, activation='softmax', name='output02')(x)  # 输出189个数
        
        output03 = layers.Dense(2, activation='softmax', name='output03')(x)  # 输出2个数
        output04 = layers.Dense(15, activation='softmax', name='output04')(x)  # 输出15个数
        output05 = layers.Dense(15, activation='softmax', name='output05')(x)  # 输出15个数
        output06 = layers.Dense(15, activation='softmax', name='output06')(x)  # 输出15个数
        output07 = layers.Dense(15, activation='softmax', name='output07')(x)  # 输出15个数
        output08 = layers.Dense(15, activation='softmax', name='output08')(x)  # 输出15个数
        output09 = layers.Dense(13, activation='softmax', name='output09')(x)  # 输出13个数
        output10 = layers.Dense(13, activation='softmax', name='output10')(x)  # 输出13个数
        output11 = layers.Dense(13, activation='softmax', name='output11')(x)  # 输出13个数
        output12 = layers.Dense(13, activation='softmax', name='output12')(x)  # 输出13个数

        # 创建模型
        model = models.Model(inputs=inputs, outputs=[
            output01,
            output02,
            output03,
            output04,
            output05,
            output06,
            output07,
            output08,
            output09,
            output10,
            output11,
            output12
        ])

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


