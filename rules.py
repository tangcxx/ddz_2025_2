
#%%
import numpy as np
import itertools
import pickle
import re

#%%
## 牌组及对牌编码的函数
## 一共54张
## 0 ~ 14，依次代表扑克牌的 3 ~ 大王
CARDS = np.array([12, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
                     12, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
                     12, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
                     12, 13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
                     14, 15]) - 1


## 牌的种类，牌组编码后的维度
CARD_DIM = 15


## 把一组牌的清单编码成15维的向量
def list2vec(cards):
    '''
    把一组牌的清单编码成15维的向量

    cards: 列表
        列表形式的一组牌，每个元素代表一张牌

    返回值：长度15的数组
        代表输入的那一组牌，数组的每个元素表示对应的牌的张数
    '''
    counts = np.zeros(CARD_DIM, int)  ##定义长度为CARD_DIM(即15)的数组
    for i in cards:
        counts[i] += 1  ## 计数，对于输入列表中的每一张牌，counts 对应位置的数字加1
    return counts

## 把向量变回清单
def vec2list(cards):
    '''
    list2vec的逆操作
    
    cards: 长度15的数组
        表示一组牌
    
    返回值：列表，每个元素代表一张牌
    '''
    res = []
    for i in range(len(cards)):  
        res += [i] * cards[i]  ## 将 cards[i] 张 扑克i 加到列表里
    return res

## 把清单变成文字，以便输出显示
def list2str(cards):
    '''
    把清单变成文字，以便输出显示

    cards: 列表，每个元素代表一张牌
    
    返回值：字符串
        文字形式输出一组牌，每张牌之间以空格间隔。主要用于调式时显示扑克牌内容
    '''
    cs = np.array(["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "小王", "大王"])
    return " ".join(cs[cards])

## 把向量变成文字
def vec2str(cards):
    '''
    把向量形式的一组牌变成字符串形式
    先调用 vec2list 把向量变成列表，再调用 list2str 把列表变成文字

    cards: 长度15的数组

    返回值：字符串
        文字形式输出一组牌，每张牌之间以空格间隔。主要用于调式时显示扑克牌内容
    '''
    return list2str(vec2list(cards))

## 把文字形式的一组牌变成向量
def str2vec(cards):
    '''
    vec2str的逆操作，把文字形式的一组牌变成向量
    cards: 字符串
        文字形式的一组牌，每张牌之间以空格间隔
    
    返回值：长度15的数组
        代表一组牌
    '''
    vec = np.zeros(CARD_DIM, int)
    d = {
        "3":0, "4":1, "5":2, "6":3, "7":4, "8":5, "9":6, "10":7, "J":8, 
        "Q":9, "K":10, "A":11, "2":12, "小王":13, "大王":14
    }  ## 字典，每张牌的牌面文字对应的编码
    for w in re.split(" +", cards.upper()):  ## 分割输入字符串
       if w in d:  ## 如果字符存在于字典中
            vec[d[w]] += 1  ## 找到对应编码，数组对应位置元素加1
    return vec


#%%
## 人为计算一组牌的一些特征数据，作为神经网络输入的 meta 的一部分
def ccounts(cards):
    '''
    人为计算一组牌的一些特征数据，作为神经网络输入的 meta 的一部分
    
    cards: 长度15数组，一组牌

    返回值：8个整数
        单牌数量，对子数量，三张数量，炸弹数量，单牌平均大小，对子平均大小，三张平均大小，炸弹平均大小
        向下取整
    '''
    cards = cards.copy()
    nbomb = np.sum(cards == 4)  ## 炸弹数量
    avgbomb = np.where(cards == 4)[0].mean() if nbomb else 0  ## 炸弹平均大小
    if cards[13] and cards[14]:  ## 有没有王炸，有的话修改炸弹数量和炸弹平均大小
        nbomb += 1  
        avgbomb += (14-avgbomb)/nbomb
    avgbomb = int(avgbomb)
    nthree = np.sum(cards == 3)  ## 三张数量
    avgthree = int(np.where(cards == 3)[0].mean()) if nthree else 0  ## 三张平均大小
    npair = np.sum(cards == 2)  ## 对子数量
    avgpair = int(np.where(cards == 2)[0].mean()) if npair else 0  ## 对子平均大小
    ## 下面计算单牌数量，需要排除可以当顺子出的单牌
    singles = [] ## 保存找到的单张 
    s = set()
    for h in playable_strait(cards)[0]:
        s = set.union(s, h)
    for i in np.arange(12):
        if cards[i] == 1 and i not in s:
            singles.append(i)
    if cards[12] == 1:  ## 如果有单张2
        singles.append(12)
    if cards[13] == 1 and cards[14] == 0:  ## 如果有单张小王
        singles.append(13)
    if cards[14] == 1 and cards[13] == 0:  ## 如果有单张大王
        singles.append(14)
    nsingle = len(singles)  ## 单张数量
    avgsingle = int(np.mean(singles)) if nsingle else 0  ##单张平均值
    return (nsingle, npair, nthree, nbomb, avgsingle, avgpair, avgthree, avgbomb)

## 标记某张牌是否单张，不考虑2, 小王, 大王
def singlestatus(cards):
    singles = np.zeros(CARD_DIM, int)
    s = set()
    for h in playable_strait(cards)[0]:
        s = set.union(s, h)
    for i in np.arange(12):
        if cards[i] == 1 and i not in s:
            singles[i] = 1
    return singles

#%%
## 以下计算所有合法出牌

def playable_single(cards, prerng = [-1, -1]): #单张
    res = []
    rngs = []
    mini = prerng[0] + 1
    for i in np.arange(mini, 15):
        if cards[i] > 0:
            res.append([i])
            rngs.append([i, i])
    return (res, 0, rngs)

def playable_pair(cards, prerng = [-1, -1]): #对子
    res = []
    rngs = []
    mini = prerng[0] + 1
    for i in np.arange(mini, 13):
        if cards[i] > 1:
            res.append([i, i])
            rngs.append([i, i])
    return (res, 1, rngs)

def playable_three(cards, prerng = [-1, -1]): #三个
    res = []
    rngs = []
    mini = prerng[0] + 1
    for i in np.arange(mini, 13):
        if cards[i] > 2:
            res.append([i, i, i])
            rngs.append([i, i])
    return (res, 2, rngs)

def playable_threewithone(cards, prerng = [-1, -1]): #三带一
    res = []
    rngs = []
    mini = prerng[0] + 1
    for i in np.arange(mini, 13):
        if cards[i] > 2:
            for j in range(15):
                if j != i and cards[j] > 0:
                    res.append([i, i, i, j])
                    rngs.append([i, i])
    return (res, 3, rngs)

def playable_threewithpair(cards, prerng = [-1, -1]): #三带一对
    res = []
    rngs = []
    mini = prerng[0] + 1
    for i in np.arange(mini, 13):
        if cards[i] > 2:
            for j in range(15):
                if j != i and cards[j] > 1:
                    res.append([i, i, i, j, j])
                    rngs.append([i, i])
    return (res, 4, rngs)

def playable_bomb(cards, prerng = [-1, -1]): #四个，炸弹
    res = []
    rngs = []
    mini = prerng[0] + 1
    for i in np.arange(mini, 13):
        if cards[i] > 3:
            res.append([i, i, i, i])
            rngs.append([i, i])
    return (res, 5, rngs)

def playable_jokerbomb(cards, prerng = [-1, -1]): #王炸
    if cards[13] and cards[14]:
        return ([[13, 14]], 6, [[13, 14]])
    return([], 6, [])

def playable_fourwithones(cards, prerng = [-1, -1]):  #四带二
    res = []
    rngs = []
    mini = prerng[0] + 1
    for i in np.arange(mini, 13):
        if cards[i] > 3:
            remains = cards.copy()
            remains[i] = 0
            remainlist = vec2list(remains)
            for c in set([*itertools.combinations(remainlist, 2)]):
                if not (13 in c and 14 in c):
                    res.append([i, i, i, i] + list(c))
                    rngs.append([i, i])
    return (res, 7, rngs)

def playable_fourwithpairs(cards, prerng = [-1, -1]): #四带两对
    res = []
    rngs = []
    mini = prerng[0] + 1
    for i in np.arange(mini, 13):
        if cards[i] > 3:
            pairlist = []
            for j in range(13):
                if j != i and cards[j] > 1:
                    pairlist.append(j)
            for c in itertools.combinations(pairlist, 2):
                res.append([i, i, i, i] + list(c) * 2)
                rngs.append([i, i])
    return (res, 8, rngs)

def strait_length(cards, length, mini):
    res = []
    rngs = []
    i = mini
    j = i + length
    if cards[i:j].all():
        res.append([k for k in np.arange(i, j)])
        rngs.append([i, j-1])
    return (res, rngs)

def playable_strait(cards, prerng = [-1, -1]): #顺子
    res = []
    rngs = []
    if prerng[0] == -1:
        for length in np.arange(5, 13):
            for mini in np.arange(0, 13-length):
                re, rng = strait_length(cards, length, mini)
                res += re
                rngs += rng
    else:
        length = prerng[1] - prerng[0] + 1
        for mini in np.arange(prerng[0] + 1, 13-length):
            re, rng = strait_length(cards, length, mini)
            res += re 
            rngs += rng
    return (res, 9, rngs)

def pairs_length(cards, length, mini):
    res = []
    rngs = []
    i = mini
    j = i + length
    if np.all(cards[i:j] > 1):
        tmp = [k for k in np.arange(i, j)]
        res.append(tmp * 2)
        rngs.append([i, j-1])
    return (res, rngs)
        
def playable_pairs(cards, prerng = [-1, -1]): #连对
    res = []
    rngs = []
    if prerng[0] == -1:
        for length in np.arange(3, 11):
            for mini in np.arange(0, 13-length):
                re, rng = pairs_length(cards, length, mini)
                res += re 
                rngs += rng 
    else:
        length = prerng[1] - prerng[0] + 1
        for mini in np.arange(prerng[0] + 1, 13-length):
            re, rng = pairs_length(cards, length, mini)
            res += re 
            rngs += rng
    return (res, 10, rngs)

def plane_length(cards, length, mini):
    res = []
    rngs = [] 
    i = mini
    j = i + length
    if np.all(cards[i:j] > 2):
        tmp = [k for k in np.arange(i, j)]
        res.append(tmp * 3)
        rngs.append([i, j-1])
    return res, rngs 

def playable_plane(cards, prerng = [-1, -1]): #飞机
    res = []
    rngs = []
    if prerng[0] == -1:
        for length in np.arange(2, 7):
            for mini in np.arange(0, 13-length):
                re, rng = plane_length(cards, length, mini)
                res += re
                rngs += rng 
    else:
        length = prerng[1] - prerng[0] + 1
        for mini in np.arange(prerng[0] + 1, 13-length):
            re, rng = plane_length(cards, length, mini)
            res += re 
            rngs += rng
    return (res, 11, rngs)

def planewithones_length(cards, length, mini):
    res = []
    rngs = []
    i = mini
    j = i + length
    if np.all(cards[i:j] > 2):
        tmp = [k for k in np.arange(i, j)]
        plane = tmp * 3
        remains = cards.copy()
        for k in np.arange(i, j):
            remains[k] -=3
        remainslist = vec2list(remains)
        combins = set([*itertools.combinations(remainslist, length)])
        for addi in combins:
            if sum([i <= r < j for r in addi]) >= 2:
                continue
            elif 13 in addi and 14 in addi:
                continue
            elif np.any(list2vec(addi) > 2):
                continue
            res.append(plane + list(addi)) 
            rngs.append([i, j-1])            
    return (res, rngs)

def playable_planewithones(cards, prerng = [-1, -1]): #飞机带单张
    res = []
    rngs = []
    if prerng[0] == -1:
        for length in np.arange(2, 6):
            for mini in np.arange(0, 13-length):
                re, rng = planewithones_length(cards, length, mini)
                res += re
                rngs += rng 
    else:
        length = prerng[1] - prerng[0] + 1
        for mini in np.arange(prerng[0] + 1, 13-length):
            re, rng = planewithones_length(cards, length, mini)
            res += re 
            rngs += rng 
    return (res, 12, rngs)
        
def planewithpairs_length(cards, length, mini):
    res = []
    rngs = []
    i = mini
    j = i + length
    if np.all(cards[i:j] > 2):
        tmp = [k for k in np.arange(i, j)]
        plane = tmp * 3
        remains = cards.copy()
        for k in np.arange(i, j):
            remains[k] -=3
        pairs = []
        for k in range(13):
            if remains[k] > 1:
                pairs.append(k)
        if len(pairs) >= length:
            for addi in [*itertools.combinations(pairs, length)]:
                res.append(plane + list(addi) * 2) 
                rngs.append([i, j-1])
    return (res, rngs)

def playable_planewithpairs(cards, prerng = [-1, -1]):  #飞机带对子
    res = []
    rngs = []
    if prerng[0] == -1:
        for length in np.arange(2, 5):
            for mini in np.arange(0, 13-length):
                re, rng = planewithpairs_length(cards, length, mini)
                res += re
                rngs += rng 
    else:
        length = prerng[1] - prerng[0] + 1
        for mini in np.arange(prerng[0] + 1, 13-length):
            re, rng = planewithpairs_length(cards, length, mini)
            res += re 
            rngs += rng 
    return (res, 13, rngs)

def playable_void(cards, prerng = [-1, -1]): #过
    return ([[]], 999, [[-1, -1]])

playables = [playable_single, playable_pair, playable_three, 
             playable_threewithone, playable_threewithpair, playable_bomb, 
             playable_jokerbomb, playable_fourwithones, playable_fourwithpairs, 
             playable_strait, playable_pairs, playable_plane,
             playable_planewithones, playable_planewithpairs]

#%%
def playable0(cards, *argt, **argd):  #主手
    res = []
    typs = []
    rngs = []
    for f in playables:
        re, typ, rng = f(cards)
        res += re
        typs += [typ] * len(re)
        rngs += rng
    return (res, typs, rngs)

def playable1(cards, precards, pretyp, prerng):  #应手
    res, typs, rngs = playable_void(None)
    typs = [typs]
    if pretyp == 6: #对方王炸
        return (res, typs, rngs)
    re, typ, rng = playables[pretyp](cards, prerng)
    res += re
    typs += [typ] * len(re)
    rngs += rng
    if pretyp != 5: #找炸弹
        re, typ, rng = playable_bomb(cards, [-1, -1])
        res += re
        typs += [typ] * len(re)
        rngs += rng
    re, typ, rng = playable_jokerbomb(cards) #王炸
    res += re
    typs += [typ] * len(re)
    rngs += rng
    
    return (res, typs, rngs)

playable = [playable0, playable1]

def withoutsidekick(hand):
    typ, rng = ap[tuple(hand)]
    cards = np.zeros(CARD_DIM, int)
    if typ == 3 or typ == 4: #三带1 三带2
        cards[rng[0]] = 3
    elif typ == 7 or typ == 8: #四带2 四带两对
        cards[rng[0]] = 4
    elif typ == 12 or typ == 13: #飞机带单张 飞机带对子
        for i in np.arange(rng[0], rng[1]+1):
            cards[i] = 3
    else:
        return hand
    return cards
    

# allplayable = playable[0](np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1]))
# allplayable_k = [tuple(list2vec(i)) for i in allplayable[0]] + [(0,)*15]
# allplayable_v = [*zip(allplayable[1], allplayable[2])] + [(999, [-1, -1])]
# ap = dict(zip(allplayable_k, allplayable_v))
# with open("ap.pkl", "wb") as f:
#     pickle.dump(ap, f)

## 字典，记录每手合法出牌对应的牌型和大小
with open("ap.pkl", "rb") as f:
    ap = pickle.load(f)

# %%
