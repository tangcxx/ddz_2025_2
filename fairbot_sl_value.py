#%% 
import numpy as np 
import torch
from torch import nn
import rules

X_LEN = 403

onehot = {
    0: np.array([0, 0, 0, 0], np.int8),
    1: np.array([1, 0, 0, 0], np.int8),
    2: np.array([1, 1, 0, 0], np.int8),
    3: np.array([1, 1, 1, 0], np.int8),
    4: np.array([1, 1, 1, 1], np.int8)
}

def vec2onehot(vec):
    res = np.zeros(54, np.int8)
    res[52:] = (vec[13:] == 1)
    for i in range(13):
        res[i*4:i*4+4] = onehot[vec[i]]
    return res

def list2onehot(cards):
    return vec2onehot(rules.list2vec(cards))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        
        self.fc1 = nn.Linear(128+X_LEN, 512)
        self.norm1 = nn.LayerNorm(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(512, 512)
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

    def forward(self, z, x):
        lstm_out, (h_n, _) = self.lstm(z)
        h_n = h_n.squeeze(0)
        x = torch.cat([h_n,x], dim=-1)
        
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.norm4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

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
        self.zs = []
        self.xs_cheating = []

    def initiate(self, arena, role):
        self.arena = arena
        self.xs = []
        self.zs = []
        self.xs_cheating = []

    def play(self):
        arena = self.arena
        choices = arena.getChoices()
        return self.netchoose(arena, choices)

    def update(self):
        pass

# 输入数据
# 出牌历史
# 底牌 54
# 其他两家手牌 54
# 另两家最近一手出牌 2*54
# 另两家乘余手牌 20 + 20
# 我的手牌 54
# 我的出牌 54

    def getdata(self, arena, choices):
        x = np.zeros(X_LEN, dtype=np.int8)
        
        remain, lastplay = arena.remain, arena.lastplay
        pos_cur, b1, b2 = arena.pos, arena.b1, arena.b2
        
        x[0:54] = list2onehot(arena.cards[0:3]) # 底牌
        x[54:108] = vec2onehot(remain[b1] + remain[b2]) # 其他两家手牌
        x[108:162] = vec2onehot(lastplay[b1]) # 上家上一手出牌
        x[162:216] = vec2onehot(lastplay[b2]) # 下家上一手出牌
        remain_len_b1, remain_len_b2 = np.sum(remain[b1]), np.sum(remain[b2])
        if remain_len_b1 > 0:
            x[215+remain_len_b1] = 1 # 上家手牌数量
        if remain_len_b2 > 0:
            x[235+remain_len_b2] = 1 # 下家手牌数量
        
        xs = np.repeat(x[np.newaxis, :], len(choices), axis=0)
        
        for idx_choice, choice in enumerate(choices):
            remain_cur = remain[pos_cur] - choice
            xs[idx_choice, 256:310] = vec2onehot(remain_cur)  # 我的手牌
            xs[idx_choice, 310:364] = vec2onehot(choice) # 出牌
            
            for i in range(8):
                if np.all(remain_cur[i:(i+5)] >= 1):
                    xs[idx_choice, (364+i):(369+i)] = 1           ## 顺子 364:377
            for i in range(10):
                if np.all(remain_cur[i:(i+3)] >= 2):
                    xs[idx_choice, (377+i):(380+i)] = 1           ## 连对 377:390  
            for i in range(11):
                if np.all(remain_cur[i:(i+2)] >= 3):
                    xs[idx_choice, (390+i):(392+i)] = 1           ## 飞机 390:403

            
        z = np.zeros((30, 54), dtype=np.int8)  # 前30手出牌历史
        hs = arena.history[-30:]
        for i in range(30-len(hs), 30):
            z[i, :] = vec2onehot(hs[i - (30-len(hs))])
        z = z.reshape(10, 162)
        zs = np.repeat(z[np.newaxis, :], len(choices), axis=0)
        
        zs = torch.from_numpy(zs).float()
        xs = torch.from_numpy(xs).float()
            
        return zs, xs

# 0,1,2,3层: 手牌
    # 4,5,6,7: 上一轮出牌
    # 8: 单张
    # 9: 对子
    # 10: 三张
    # 11：炸弹
    # 12：顺子
    # 13: 连对
    # 14: 飞机

    def getdata_cheating(self, arena, choices):

        def getdata_pos(remain, lastplay, output):
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
        
        xs = np.zeros((15, 3, 15), np.int8)
        remain, lastplay = arena.remain, arena.lastplay
        pos_cur, b1, b2 = arena.pos, arena.b1, arena.b2

        getdata_pos(remain[b1], lastplay[b1], xs[:, b1, :])
        getdata_pos(remain[b2], np.zeros(15), xs[:, b2, :])
        
        xs = np.repeat(xs[np.newaxis, :], len(choices), axis=0)
        
        for idx_choice, choice in enumerate(choices):
            remain_cur = remain[pos_cur] - choice
            lastplay_cur = choice
            getdata_pos(remain_cur, lastplay_cur, xs[idx_choice, :, pos_cur, :])
        xs = torch.from_numpy(xs).float()
        return xs


    def netchoose(self, arena, choices):
        ## 调用 get_dizhu_win_probs 计算choices里每种出牌后的局面估值
        zs, xs, scores = self.get_dizhu_win_probs(arena, choices)
        if self.arena.pos != 0:
            scores = 1 - scores
        idx = np.argmax(scores)  ## 找到最大的那一项的序号（下标）
        if self.verbos & 1:
            for i in np.argsort(scores)[::-1][0:5]:
                print(scores[i], rules.vec2str(choices[i]))

        self.zs.append(zs[idx:(idx+1)])
        self.xs.append(xs[idx:(idx+1)])  
        self.xs_cheating.append(self.getdata_cheating(arena, choices[idx:idx+1]))
        return choices[idx]

    def get_dizhu_win_probs(self, arena, choices):
        zs, xs = self.getdata(arena, choices)
        model = self.models[arena.pos]
        model.eval()
        with torch.no_grad():
            preds = model(zs, xs).numpy()[:,0]
        return zs, xs, preds

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
