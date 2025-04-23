#%% 
import numpy as np 
import torch
from torch import nn
import rules

X_LEN = 364

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
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        
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

    def initiate(self, arena, role):
        self.arena = arena
        self.xs = []
        self.zs = []

    def play(self):
        arena = self.arena
        choices = arena.getChoices()
        return self.netchoose(arena, choices)

    def update(self):
        pass

# 输入数据
# 出牌历史
# 底牌 54
# 我的手牌 54
# 其他两家手牌 54
# 另两家最近一手出牌 2*54
# 另两家乘余手牌 20 + 20

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
            
        z = np.zeros((15, 54), dtype=np.int8)  # 前15手出牌历史
        hs = arena.history[-15:]
        for i in range(15-len(hs), 15):
            z[i, :] = vec2onehot(hs[i - (15-len(hs))])
        z = z.reshape(5, 162)
        zs = np.repeat(z[np.newaxis, :], len(choices), axis=0)
        
        zs = torch.from_numpy(zs).float()
        xs = torch.from_numpy(xs).float()
            
        return zs, xs

    def netchoose(self, arena, choices):
        if np.random.random() < self.epsilon:
            idx = np.random.choice(len(choices))
            zs, xs = self.getdata(arena, choices[idx:idx+1])
            self.zs.append(zs[idx:(idx+1)])
            self.xs.append(xs[idx:(idx+1)])
            return choices[idx]

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
