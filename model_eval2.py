# 某种训练方法训练的模型，挑一些出来对战，选出高胜率的

import multiprocessing as mp
from datetime import datetime

import numpy as np
import keras as k
import rules
import arena as arn
ARENA = arn.ARENA

# base4: 
nround = 50
nums = []
len_segment = 80
model_path = 'model_base4'
from bot_base4 import BOT

## 以下是 model_eval2 的代码，用于继续评估模型
def model_eval_worker2(args):
    num1, num2 = args
    model1 = k.models.load_model("{}/m{}.keras".format(model_path, num1))
    model2 = k.models.load_model("{}/m{}.keras".format(model_path, num2))
    bot1 = BOT(model1, verbos=0)
    bot2 = BOT(model2, verbos=0)

    n_bot1_dizhu_win, n_bot2_dizhu_win = 0, 0
    for _ in range(nround):
        cards = rules.CARDS.copy()
        np.random.shuffle(cards)
        
        arena = ARENA(verbos=0, cards=cards.copy())
        arena.registerbot([bot1, bot2, bot2])
        arena.wholegame()
        n_bot1_dizhu_win += (arena.winner == 0)
        
        arena2 = ARENA(verbos=0, cards=cards.copy())
        arena2.registerbot([bot2, bot1, bot1])
        arena2.wholegame()
        n_bot2_dizhu_win += (arena2.winner == 0)
        
        sd = (nround * 2 * 0.5 * 0.5) ** 0.5
        d = n_bot1_dizhu_win - n_bot2_dizhu_win
    return nround, sd, num1, num2, n_bot1_dizhu_win, n_bot2_dizhu_win, d

def model_eval2():
    import itertools as it
    mp.set_start_method('spawn')

    matches = [*it.combinations(nums, 2)]
    matches_segs = [matches[i:i + len_segment] for i in range(0, len(matches), len_segment)]

    f = open('{}/eval.txt'.format(model_path), 'a', buffering=1)
    with mp.Pool(8) as p:
        for matches_seg in matches_segs:
            res = p.map(model_eval_worker2, matches_seg)
            res = np.array(res)
            for r in res:
                f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(*r))
            print(datetime.now(), matches_seg)
    f.close()
    return res

if __name__ == '__main__':
    res = model_eval2()

