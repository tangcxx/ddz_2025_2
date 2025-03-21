## 诚实AI, 基于bot_three, 随机猜牌

#%% 
import numpy as np 
import keras as k 
import rules
import bot_three

to_categorical = k.utils.to_categorical

NCARDGROUPS = 3  ##神经网络输入牌组数量
CARD_DIM = rules.CARD_DIM  ## 牌组长度，15
NCHANNEL = 15

##使用神经网络出牌的作弊机器人，它知道其他玩家手上的牌
## BOT类是演示用的，供ARENA类调用
## ARENA类要求所有BOT有如下4个成员函数
class BOT():
    def __init__(self, bots_cheating, verbos=0, sample=1):
        self.bots_cheating = bots_cheating
        self.verbos = verbos
        self.sample = sample
        self.xs = []

    def initiate(self, arena, role):
        self.arena = arena
        self.xs = []

    def play(self):
        choices = self.arena.getChoices()
        arena = self.arena
        scores = np.zeros(len(choices))
        for _ in range(self.sample):
            arena_resample = self.get_arena_resample(arena)
            __, dizhu_win_probs = self.bots_cheating[arena.pos].get_dizhu_win_probs(arena_resample, choices)
            scores += dizhu_win_probs
        if arena.pos != 0:
            scores = self.sample - scores
        if self.verbos & 1:
            self.showChoices(choices, scores, 5)
        return choices[np.argmax(scores)]

    def update(self):
        pass

    def get_arena_resample(self, arena):
        arena_resample = arena.copy()
        b1, b2 = arena.b1, arena.b2
        len1 = arena.remain[b1].sum()
        remain = arena_resample.remain
        remain_others_list = rules.vec2list(remain[b1] + remain[b2])
        np.random.shuffle(remain_others_list)
        remain[b1] = rules.list2vec(remain_others_list[0:len1])
        remain[b2] = rules.list2vec(remain_others_list[len1:])
        return arena_resample
    
    ## 调试用，打印某个局面下所有的出牌选择及相应的估值
    def showChoices(self, choices, scores, NUM=None):
        sorted_idx = np.argsort(scores)[::-1]
        if NUM:
            sorted_idx = sorted_idx[0:NUM]
        for i in sorted_idx:
            print(scores[i], rules.vec2str(choices[i]))



# %%
