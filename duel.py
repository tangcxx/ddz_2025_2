# 以 "model_base2/m{}.keras".format(386150) 为基准，评估不同训练方法的表现

import multiprocessing as mp
from datetime import datetime

import numpy as np
import keras as k
import rules
import arena as arn
ARENA = arn.ARENA

# config
nproc = 8
nround = 1000

import bot_lr
path1, id1 = "model_lr2", 191000
model1 = k.models.load_model("{}/m{}.keras".format(path1, id1))
bot1 = bot_lr.BOT(model1, verbos=0)

import bot_aug
path2, id2 = "e:/ddz_2025_2_model/model_aug", 488750
model2 = k.models.load_model("{}/m{}.keras".format(path2, id2))
bot2 = bot_aug.BOT(model2, verbos=0)

def model_eval_worker(args):
    n_bot1_dizhu_win, n_bot2_diuzhu_win = 0, 0

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

    with mp.Pool(nproc) as p:
        res = p.map(model_eval_worker, [0] * nround)
        res = np.array(res)
        res = np.sum(res, axis=0)
        print("{} {} {} {} {} {} {}".format(datetime.now(), path1, id1, path2, id2, *res))

if __name__ == '__main__':
    res = model_eval()

