# https://github.com/kwai/DouZero 移植

#%%
import numpy as np
import torch
from torch import nn

import rules

class LandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(373 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        return x
        # if return_value:
        #     return dict(values=x)
        # else:
        #     if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
        #         action = torch.randint(x.shape[0], (1,))[0]
        #     else:
        #         action = torch.argmax(x,dim=0)[0]
        #     return dict(action=action)

class FarmerLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(484 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        return x
        # if return_value:
        #     return dict(values=x)
        # else:
        #     if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
        #         action = torch.randint(x.shape[0], (1,))[0]
        #     else:
        #         action = torch.argmax(x,dim=0)[0]
        #     return dict(action=action)

weights = [torch.load("e:/DouZero-main/baselines/douzero_WP/landlord.ckpt", map_location='cpu'),
           torch.load("e:/DouZero-main/baselines/douzero_WP/landlord_down.ckpt", map_location='cpu'),
           torch.load("e:/DouZero-main/baselines/douzero_WP/landlord_up.ckpt", map_location='cpu')]

models = [LandlordLstmModel(), FarmerLstmModel(), FarmerLstmModel()]

for pos in range(3):
    models[pos].load_state_dict(weights[pos])
    
NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

def vec2onehot(vec):
    onehot = np.zeros(54, dtype=np.int8)
    for i in range(13):
        onehot[i*4:i*4+4] = NumOnes2Array[vec[i]]
    onehot[52] = vec[13] == 1
    onehot[53] = vec[14] == 1
    return onehot

#%%
class BOT:
    def __init__(self, verbos=0):
        self.verbos = verbos

    def initiate(self, arena, role):
        self.arena = arena
    
    def play(self):
        choices = self.arena.getChoices()
        return self.netchoose(choices)

    def update(self):
        pass

# x_batch = np.hstack((my_handcards_batch,
#                      other_handcards_batch,
#                      last_action_batch,
#                      landlord_up_played_cards_batch,
#                      landlord_down_played_cards_batch,
#                      landlord_up_num_cards_left_batch,
#                      landlord_down_num_cards_left_batch,
#                      bomb_num_batch,
#                      my_action_batch))


    def getdata_0(self, arena, choices):
        z, x = np.zeros((5, 162), dtype=np.int8), np.zeros(373, dtype=np.int8)
        
        x[:54] = vec2onehot(arena.remain[0]) # 我的手牌
        
        remain_others = arena.remain[1] + arena.remain[2] # 其他人的手牌
        x[54:108] = vec2onehot(remain_others)
        
        if len(arena.history) == 0:
            lastmove = np.zeros(15, dtype=np.int8)
        elif np.all(arena.history[-1] == 0):
            lastmove = arena.history[-2]
        else:
            lastmove = arena.history[-1]
        x[108:162] = vec2onehot(lastmove) # 牌桌上的上一次出牌(当前要回应的牌)
        
        history_pos2 = arena.init[2] - arena.remain[2]
        x[162:216] = vec2onehot(history_pos2) # 农民2出过的牌
        
        history_pos1 = arena.init[1] - arena.remain[1]
        x[216:270] = vec2onehot(history_pos1) # 农民1出过的牌
        
        num_left_cards_2 = np.sum(arena.remain[2])
        x[270 + num_left_cards_2 - 1] = 1  # 农民2手牌数量
        
        num_left_cards_1 = np.sum(arena.remain[1])
        x[287 + num_left_cards_1 - 1] = 1  # 农民1手牌数量
        
        bomb_num = 0
        for h in arena.history:
            if np.sum(h) == 4 and np.any(h == 4):
                bomb_num += 1
            elif h[13] == 1 and h[14] == 1:
                bomb_num += 1
        x[304 + bomb_num] = 1  # 已经打出的炸弹数量
        
        xs = np.repeat(x[np.newaxis, :], len(choices), axis=0)  # repeat
        
        for i, choice in enumerate(choices):
            xs[i, 319:373] = vec2onehot(choice)  # 我本次要出的牌

        z = np.zeros((15, 54), dtype=np.int8)  # 前15手出牌历史
        hs = arena.history[-15:]
        for i in range(15-len(hs), 15):
            z[i, :] = vec2onehot(hs[i - (15-len(hs))])
        z = z.reshape(5, 162)
        zs = np.repeat(z[np.newaxis, :], len(choices), axis=0)

        return zs, xs
    
    def getdata_1(self, arena, choices):
        z, x = np.zeros((5, 162), dtype=np.int8), np.zeros(484, dtype=np.int8)
        
        x[:54] = vec2onehot(arena.remain[1]) # 我的手牌
        
        remain_others = arena.remain[2] + arena.remain[0] # 其他人的手牌
        x[54:108] = vec2onehot(remain_others)
        
        history_pos0 = arena.init[0] - arena.remain[0]
        x[108:162] = vec2onehot(history_pos0) # 地主出过的牌
        
        history_pos2 = arena.init[2] - arena.remain[2]
        x[162:216] = vec2onehot(history_pos2) # 农民2出过的牌
        
        if np.all(arena.history[-1] == 0):
            lastmove = arena.history[-2]
        else:
            lastmove = arena.history[-1]
        x[216:270] = vec2onehot(lastmove) # 牌桌上的上一次出牌(当前要回应的牌)

        lastmove_0 = arena.history[-1]
        x[270:324] = vec2onehot(lastmove_0) # 地主上一次出牌

        lastmove_2 = arena.history[-2] if len(arena.history) >= 2 else np.zeros(15, dtype=np.int8)
        x[324:378] = vec2onehot(lastmove_2) # 农民2上一次出牌

        num_left_cards_0 = np.sum(arena.remain[0])
        x[378 + num_left_cards_0 - 1] = 1 # 地主手牌数量
        
        num_left_cards_2 = np.sum(arena.remain[2])
        x[398 + num_left_cards_2 - 1] = 1  # 农民2手牌数量
        
        bomb_num = 0
        for h in arena.history:
            if np.sum(h) == 4 and np.any(h == 4):
                bomb_num += 1
            elif h[13] == 1 and h[14] == 1:
                bomb_num += 1
        x[415 + bomb_num] = 1  # 已经打出的炸弹数量
        
        xs = np.repeat(x[np.newaxis, :], len(choices), axis=0)  # repeat
        
        for i, choice in enumerate(choices):
            xs[i, 430:484] = vec2onehot(choice)  # 我本次要出的牌

        z = np.zeros((15, 54), dtype=np.int8)  # 前15手出牌历史
        hs = arena.history[-15:]
        for i in range(15-len(hs), 15):
            z[i, :] = vec2onehot(hs[i - (15-len(hs))])
        z = z.reshape(5, 162)
        zs = np.repeat(z[np.newaxis, :], len(choices), axis=0)

        return zs, xs

    def getdata_2(self, arena, choices):
        z, x = np.zeros((5, 162), dtype=np.int8), np.zeros(484, dtype=np.int8)
        
        x[:54] = vec2onehot(arena.remain[arena.pos]) # 我的手牌
        
        remain_others = arena.remain[0] + arena.remain[1] # 其他人的手牌
        x[54:108] = vec2onehot(remain_others)
        
        history_pos0 = arena.init[0] - arena.remain[0]
        x[108:162] = vec2onehot(history_pos0) # 地主出过的牌
        
        history_pos1 = arena.init[1] - arena.remain[1]
        x[162:216] = vec2onehot(history_pos1) # 农民1出过的牌
        
        if np.all(arena.history[-1] == 0):
            lastmove = arena.history[-2]
        else:
            lastmove = arena.history[-1]
        x[216:270] = vec2onehot(lastmove) # 牌桌上的上一次出牌(当前要回应的牌)

        lastmove_0 = arena.history[-2]
        x[270:324] = vec2onehot(lastmove_0) # 地主上一次出牌

        lastmove_1 = arena.history[-1]
        x[324:378] = vec2onehot(lastmove_1) # 农民1上一次出牌

        num_left_cards_0 = np.sum(arena.remain[0])
        x[378 + num_left_cards_0 -1] = 1  # 地主手牌数量
        
        num_left_cards_1 = np.sum(arena.remain[1])
        x[398 + num_left_cards_1 - 1] = 1  # 农民1手牌数量
        
        bomb_num = 0
        for h in arena.history:
            if np.sum(h) == 4 and np.any(h == 4):
                bomb_num += 1
            elif h[13] == 1 and h[14] == 1:
                bomb_num += 1
        x[415 + bomb_num] = 1  # 已经打出的炸弹数量
        
        xs = np.repeat(x[np.newaxis, :], len(choices), axis=0)  # repeat
        
        for i, choice in enumerate(choices):
            xs[i, 430:484] = vec2onehot(choice)  # 我本次要出的牌

        z = np.zeros((15, 54), dtype=np.int8)  # 前15手出牌历史
        hs = arena.history[-15:]
        for i in range(15-len(hs), 15):
            z[i, :] = vec2onehot(hs[i - (15-len(hs))])
        z = z.reshape(5, 162)
        zs = np.repeat(z[np.newaxis, :], len(choices), axis=0)

        return zs, xs

    def getdata(self, arena, choices):
        f = [self.getdata_0, self.getdata_1, self.getdata_2]
        zs, xs = f[arena.pos](arena, choices)
        zs = torch.tensor(zs, dtype=torch.float32)
        xs = torch.tensor(xs, dtype=torch.float32)
        return zs, xs

    def netchoose(self, choices):
        if len(choices) == 1:
            return choices[0]
        
        pos = self.arena.pos
        model = models[pos]
        zs, xs = self.getdata(self.arena, choices)
        with torch.no_grad():
            ys = model(zs, xs).numpy()[:,0]

        if self.verbos & 1:
            for i in np.argsort(ys)[::-1][0:5]:
                print(ys[i], rules.vec2str(choices[i]))

        return choices[np.argmax(ys)]

