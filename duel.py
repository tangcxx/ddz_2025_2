# 以 "model_base2/m{}.keras".format(386150) 为基准，评估不同训练方法的表现

import multiprocessing as mp
from datetime import datetime

import numpy as np
import keras as k
import rules
import arena as arn
ARENA = arn.ARENA

# config
from bot_aug import BOT
model_path = 'model_aug'
nround = 1000

matches = [(392900,393200)]
duelists = []
for match in matches:
    num1, num2 = match

    model1 = k.models.load_model("{}/m{}.keras".format(model_path, num1))
    bot1 = BOT(model1)

    model2 = k.models.load_model("{}/m{}.keras".format(model_path, num2))
    bot2 = BOT(model2)
    
    duelists.append((bot1, bot2))

def model_eval_worker(ith_match):
    n_bot1_dizhu_win, n_bot2_diuzhu_win = 0, 0
    bot1, bot2 = duelists[ith_match]

    cards = rules.CARDS.copy()
    np.random.shuffle(cards)
    
    arena = ARENA(verbos=0, cards=cards.copy())
    arena.registerbot([bot1, bot2, bot2])
    arena.wholegame()
    n_bot1_dizhu_win += (arena.winner == 0)
    
    arena2 = ARENA(verbos=0, cards=cards.copy())
    arena2.registerbot([bot2, bot1, bot1])
    arena2.wholegame()
    n_bot2_diuzhu_win += (arena2.winner == 0)
    
    return n_bot1_dizhu_win, n_bot2_diuzhu_win


def model_eval():
    mp.set_start_method('spawn')

    with mp.Pool(8) as p:
        for i in range(len(duelists)):
            res = p.map(model_eval_worker, [i] * nround)
            res = np.array(res)
            res = np.sum(res, axis=0)
            print("{} {} {} {} {}".format(datetime.now(), *matches[i], *res))

if __name__ == '__main__':
    res = model_eval()

