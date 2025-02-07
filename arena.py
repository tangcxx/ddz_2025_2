#%%
import numpy as np 
import re
import keras as k

import rules

to_categorical = k.utils.to_categorical
NCARDGROUPS = 6  ##神经网络输入牌组数量
CARD_DIM = rules.CARD_DIM  ## 牌组长度，15
NCARDSPARAM = NCARDGROUPS * CARD_DIM  ##代表牌组的输入的长度
NCARDBIT = 5  ## 每种牌（牌组的每一项）用几位表示
NCHANNEL = 8

#%%
## AI竞技场
class ARENA:
    ## 洗牌
    def shuffled(self):
        '''
        返回值：长度54的数组
            洗牌，返回随机顺序的54张牌
        '''
        cards = rules.CARDS.copy()  ## 复制cards副本
        np.random.shuffle(cards)  ## 随机排序
        return cards

    ## 初始化发牌和桌面状态
    def __init__(self, verbos=0, RECORD=False, cards=None):
        self.cards = self.shuffled() if cards is None else cards
        cards = self.cards
        self.verbos = verbos
        self.init = np.array([rules.list2vec(cards[0:20]), 
                              rules.list2vec(cards[20:37]), 
                              rules.list2vec(cards[37:54])], int)
        self.remain = self.init.copy()
        self.lastplay = np.zeros((3, 15), int)
        self.pos = 0
        self.b1 = 2
        self.b2 = 1
        self.round = 0
        self.bot = []
        self.gameover = False
        self.winner = None
        self.history = []
        self.RECORD = RECORD
        self.records = []
        if self.RECORD:
            self.records.append(self.copy())
    
    def copy(self, verbos=0, RECORD=False):
        cp = object.__new__(type(self))
        cp.cards = self.cards.copy()
        cp.remain = self.remain.copy()
        cp.lastplay = self.lastplay.copy()
        cp.pos = self.pos
        cp.b1 = self.b1
        cp.b2 = self.b2
        cp.round = self.round
        cp.gameover = self.gameover
        cp.winner = self.winner
        cp.verbos = verbos
        cp.history = self.history.copy()
        cp.RECORD = RECORD 
        cp.records = []
        if cp.RECORD:
            cp.records.append(cp.copy())
        ## bot 不复制
        return cp
    
    ## 注册bot，
    ## bots是含有3个bot的列表，分别扮演地主和两个农民（同出牌顺序）
    ## 每个bot是一个类的对象，必须有一些特定的成员函数（下面有几个不同的bot类的定义）
    def registerbot(self, bots):
        self.bot = bots
        for i in range(3):
            bot = bots[i]
            bot.initiate(self, i)
    
    ## 出牌，内部调用bot方法
    ## 出牌后更新桌面状态
    def play(self, cards=None):
        bot = self.bot[self.pos]
        # choices = self.getChoices()
        # cards = bot.play(choices)
        cards = bot.play() if cards is None else cards
        self.update(cards)
        if self.verbos & 1:
            print("{0: >2d}".format(self.round-1), ":", [' 地主', '农民1', '农民2'][self.b1], ":", rules.vec2str(self.lastplay[self.b1]), "|", rules.vec2str(self.remain[self.b1]))
        for bot in self.bot:
            bot.update()
    
    ## 出牌及更新桌面状态
    def update(self, cards):
        self.remain[self.pos] -= cards
        self.lastplay[self.pos] = cards
        if self.remain[self.pos].sum() == 0:
            self.gameover = True
            self.winner = self.pos
        self.pos, self.b1, self.b2 = self.b2, self.pos, self.b1
        self.round += 1
        if self.RECORD:
            self.records.append(self.copy())
        self.history.append(cards)

    def getChoices(self):
        '''
        返回值：一个list，其中每一项是长度15的数组。
            每一项代表一种合法的出牌。空过(如果合法)也包含在内。
        '''
        ## 如果上家出牌数量不为0，根据上家的上一手出牌计算当前玩家的合法出牌
        if self.lastplay[self.b1].sum() != 0:
            ##取上家上一手牌的牌型和大小
            pretyp, prerng = rules.ap[tuple(self.lastplay[self.b1])] 
            ##计算当前玩家所有合法出牌
            choices, _, _ = rules.playable[1](self.remain[self.pos], None, pretyp, prerng)  
        
        ## 否则，根据上上家（也就是下家）的上一手出牌计算当前玩家的合法出牌
        elif self.lastplay[self.b2].sum() != 0:
            ##取上家上一手牌的牌型和大小
            pretyp, prerng = rules.ap[tuple(self.lastplay[self.b2])]
            ##计算当前玩家所有合法出牌
            choices, _, _ = rules.playable[1](self.remain[self.pos], None, pretyp, prerng)
        
        ## 否则，当前玩家主手，自主出牌
        else:
            choices, _, _ = rules.playable[0](self.remain[self.pos])
            
        choices = [rules.list2vec(choice) for choice in choices]  ## 格式转换
        return choices

    ##完整的一局
    def wholegame(self):
        while not self.gameover:
            self.play()

#%%
##纯随机出牌机器人
class BOT_RANDOM:
    def __init__(self):
        pass
    
    def initiate(self, arena, role):
        self.arena = arena
    
    def play(self):
        arena = self.arena
        choices = arena.getChoices()
        idx = np.random.choice(len(choices))
        return choices[idx]

    def update(self):
        pass

#%%
class BOT_NOIAMNOTBOT:
    def __init__(self):
        pass
    
    def initiate(self, arena, role):
        tip = '''\
            出牌格式，以空格分格，顺序随意
            例：
            三个3带对10：3 3 3 10 10
            对钩：J J
            大王：DW
            小王：XW
            '''
        print(tip)
        print(" 地主:", rules.vec2str(arena.remain[0]))
        print("农民1:", rules.vec2str(arena.remain[1]))
        print("农民2:", rules.vec2str(arena.remain[2]))
        print()
        self.arena = arena
        self.role = role
    
    def play(self):
        arena = self.arena
        choices = arena.getChoices()
        choices = {tuple(c) for c in choices}

        print("手牌:", rules.vec2str(arena.remain[arena.pos]))

        d = {
            '3': 0, '4': 1, '5': 2, '6': 3, '7': 4,
            '8': 5, '9': 6, '10': 7, 'J': 8, 'Q': 9,
            'K': 10, 'A': 11, '2': 12, 'XW': 13, 'DW': 14,
        }
        inp = input("请输入出牌: ").upper()
        while True:
            choice = np.zeros(15, int)
            inp = re.split("\\s+", inp)
            for w in inp:
                if w in d:
                    choice[d[w]] += 1
            choice = tuple(choice)
            if choice in choices:
                return choice
            else:
                inp = input("出牌错误，请重新输入: ").upper()
    
    def update(self):
        arena = self.arena
        if arena.gameover:
            if (arena.winner == 0 and self.role == 0) or (arena.winner != 0 and self.role != 0):
                print("你赢了。你保住了人类的面子，想必很自豪吧。")
            else:
                print("已经很不错了。大部分人类都达不到你的水平。")
        pass

        


#%%
bot_rd = BOT_RANDOM()
bot_me = BOT_NOIAMNOTBOT()

#%%

