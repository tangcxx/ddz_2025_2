#%% 
import numpy as np 
import torch
from torch import nn
import rules


NCARDGROUPS = 3  ##神经网络输入牌组数量
CARD_DIM = rules.CARD_DIM  ## 牌组长度，15
NCHANNEL = 15


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(15, 512, (1, 15))
        self.norm1 = nn.LayerNorm((512, 3, 1))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv2d(512, 512, (3, 1))
        self.flatten = nn.Flatten()
        self.norm2 = nn.LayerNorm(512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(512, 512)
        self.norm3 = nn.LayerNorm(512)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(512, 512)
        self.norm4 = nn.LayerNorm(512)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(512, 512)
        self.norm5 = nn.LayerNorm(512)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.2)

        self.fc6 = nn.Linear(512, 1)
        self.output = nn.Sigmoid()

    def forward(self, x):
        # Convolutional Block 1
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Convolutional Block 2
        x = self.conv2(x)
        # Flatten the output from convolutional layers
        x = self.flatten(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)


        # Fully Connected Block 1
        x = self.fc3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        # Fully Connected Block 2
        x = self.fc4(x)
        x = self.norm4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        # Fully Connected Block 3
        x = self.fc5(x)
        x = self.norm5(x)
        x = self.relu5(x)
        x = self.dropout5(x)

        x = self.fc6(x)
        return self.output(x)

##使用神经网络出牌的作弊机器人，它知道其他玩家手上的牌
## BOT类是演示用的，供ARENA类调用
## ARENA类要求所有BOT有如下4个成员函数
class BOT:
    def __init__(self, models, verbos=0, epsilon=0):
        self.models = models
        self.verbos = verbos
        self.epsilon = epsilon
        self.xs = []

    def initiate(self, arena, role):
        self.arena = arena
        self.xs = []

    def play(self):
        arena = self.arena
        choices = arena.getChoices()
        return self.netchoose(arena, choices)

    def update(self):
        pass

    # 0,1,2,3层: 手牌
    # 4,5,6,7: 上一轮出牌
    # 8: 单张
    # 9: 对子
    # 10: 三张
    # 11：炸弹
    # 12：顺子
    # 13: 连对
    # 14: 飞机
    def getdata_pos(self, remain, lastplay, output):
        output[0, :] = remain > 0
        output[1, :] = remain > 1
        output[2, :] = remain > 2
        output[3, :] = remain > 3
        output[4, :] = lastplay > 0
        output[5, :] = lastplay > 1
        output[6, :] = lastplay > 2
        output[7, :] = lastplay > 3
        output[8, :] = remain == 1
        output[9, :] = remain == 2
        output[10, :] = remain == 3
        output[11, :] = remain == 4
        for i in range(8):
            if np.all(remain[i:(i+5)] >= 1):
                output[12, i:(i+5)] = 1           ## 12
        for i in range(10):
            if np.all(remain[i:(i+3)] >= 2):
                output[13, i:(i+3)] = 1           ## 13  
        for i in range(11):
            if np.all(remain[i:(i+2)] >= 3):
                output[14, i:(i+2)] = 1           ## 14

    def getdata(self, arena, choices):
        xs = np.zeros((NCHANNEL, NCARDGROUPS, CARD_DIM), np.int8)
        remain, lastplay = arena.remain, arena.lastplay
        pos_cur, b1, b2 = arena.pos, arena.b1, arena.b2

        self.getdata_pos(remain[b1], lastplay[b1], xs[:, b1, :])
        self.getdata_pos(remain[b2], np.zeros(15), xs[:, b2, :])
        
        xs = np.repeat(xs[np.newaxis, :], len(choices), axis=0)
        
        for idx_choice, choice in enumerate(choices):
            remain_cur = remain[pos_cur] - choice
            lastplay_cur = choice
            self.getdata_pos(remain_cur, lastplay_cur, xs[idx_choice, :, pos_cur, :])
        xs = torch.from_numpy(xs).float()
        return xs

    def netchoose(self, arena, choices):
        if np.random.random() < self.epsilon:
            idx = np.random.choice(len(choices))
            self.xs.append(self.getdata(arena, choices[idx:idx+1]))
            return choices[idx]

        ## 调用 get_dizhu_win_probs 计算choices里每种出牌后的局面估值
        xs, scores = self.get_dizhu_win_probs(arena, choices)
        if self.arena.pos != 0:
            scores = 1 - scores
        idx = np.argmax(scores)  ## 找到最大的那一项的序号（下标）
        if self.verbos & 1:
            for i in np.argsort(scores)[::-1][0:5]:
                print(scores[i], rules.vec2str(choices[i]))

        self.xs.append(xs[idx:(idx+1)])  ## 记录状态
        return choices[idx]

    def get_dizhu_win_probs(self, arena, choices):
        xs = self.getdata(arena, choices)
        # xs = torch.from_numpy(xs).float()
        model = self.models[arena.pos]
        model.eval()
        with torch.no_grad():
            preds = model(xs).numpy()[:,0]
        return xs, preds

    ## 调试用，打印某个局面下所有的出牌选择及相应的估值
    def showChoices(self, arena, NUM=None):
        choices = arena.getChoices()
        _, scores = self.get_dizhu_win_probs(arena, choices)
        if arena.pos != 0:
            scores = 1 - scores
        if NUM is None:
            NUM = len(choices)
        for i in np.argsort(scores)[::-1][0:NUM]:
            print(scores[i], rules.vec2str(choices[i]))



# %%
