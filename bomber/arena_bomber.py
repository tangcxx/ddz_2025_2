#%%
import numpy as np
import rules
import arena as arn 

#%%



#%%
class ARENA(arn.ARENA):
    def shuffled(self):
        cards = rules.CARDS.copy()
        np.random.shuffle(cards)
        s = [2, 3, 6, 7, 8, 9, 11, 12]
        for i in np.arange(10):
            p1 = np.concatenate((cards[3:9], cards[21:27], cards[39:44]))
            p2 = np.concatenate((cards[9:15], cards[27:33], cards[44:49]))
            p3 = np.concatenate((cards[15:21], cards[33:39], cards[49:54]))
            cards[3:20] = np.concatenate((p1[~np.isin(p1, s)], p1[np.isin(p1, s)]))
            cards[20:37] = np.concatenate((p2[~np.isin(p2, s)], p2[np.isin(p2, s)]))
            cards[37:54] = np.concatenate((p3[~np.isin(p3, s)], p3[np.isin(p3, s)]))
            j = np.random.randint(3,54)
            cards = np.concatenate((cards[0:3], cards[j:54], cards[3:j])) 

        cardlist = list(cards[3:54])

        l = [[], [], []]
        candidates = [*range(14)]
        for i in cards[0:3]:
            if i in candidates:
                candidates.remove(i)
            elif i == 14 and 13 in candidates:
                candidates.remove(13)

        nbomb = np.random.randint(1,8)
        candidates = np.random.choice(candidates, nbomb, replace=False)
        for i in candidates:
            if i == 13:
                players = [j for j in range(3) if len(l[j]) <= 17 - 2]
                player = np.random.choice(players)
                l[player] += [13, 14]
                cardlist.remove(13)
                cardlist.remove(14)
            else:
                players = [j for j in range(3) if len(l[j]) <= 17 - 4]
                player = np.random.choice(players)
                l[player] += [i] * 4 
                for j in range(4):
                    cardlist.remove(i)

        for i in range(3):
            less = 17 - len(l[i])
            l[i] += cardlist[:less]
            cardlist = cardlist[less:]

        l[0] = list(cards[0:3]) + l[0]

        return np.array(l[0] + l[1] + l[2], int)

#     def __init__(self, verbos=0, RECORD=False):
#         cards = shuffled()
#         self.verbos = verbos
#         self.init = np.array([rules.list2vec(cards[0:20]), 
#                               rules.list2vec(cards[20:37]), 
#                               rules.list2vec(cards[37:54])], int)
#         self.remain = self.init.copy()
#         self.lastplay = np.zeros((3, 15), int)
#         self.pos = 0
#         self.b1 = 2
#         self.b2 = 1
#         self.round = 0
#         self.bot = []
#         self.gameover = False
#         self.winner = None
#         self.getdata()
#         self.RECORD = RECORD
#         self.records = []
#         if self.RECORD:
#             self.records.append(self.copy())
    
#     def copy(self, verbos=0, RECORD=False):
#         cp = object.__new__(type(self))
#         cp.remain = self.remain.copy()
#         cp.lastplay = self.lastplay.copy()
#         cp.pos = self.pos
#         cp.b1 = self.b1
#         cp.b2 = self.b2
#         cp.round = self.round
#         cp.gameover = self.gameover
#         cp.winner = self.winner
#         cp.data = self.data.copy()
#         cp.verbos = verbos
#         cp.RECORD = RECORD 
#         cp.records = []
#         if cp.RECORD:
#             cp.records.append(cp.copy())
#         ## bot 不复制
#         return cp
