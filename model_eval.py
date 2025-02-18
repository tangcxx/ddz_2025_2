import multiprocessing as mp
from datetime import datetime

import numpy as np
import keras as k
import rules
import arena as arn
import bot_base2
ARENA = arn.ARENA

model_rival = k.models.load_model("model_base2/m{}.keras".format(386150))
bot_rival = bot_base2.BOT(model_rival, verbos=0)

# 参数
# base2: 395300
maxnum = 395300
n_segment = maxnum//4000 + 1
log_file = 'model_eval_base2.txt'
model_path = 'model_base2'
from bot_base2 import BOT

# # sarsa: 93200
# maxnum = 93200
# n_segment = maxnum//4000 + 1
# log_file = 'model_eval_sarsa.txt'
# model_path = 'model_sarsa'
# from bot_sarsa import BOT

# # base3: 19800
# maxnum = 19800
# n_segment = maxnum//4000 + 1
# log_file = 'model_eval_base3.txt'
# model_path = 'model_base3'
# from bot_base3 import BOT

def model_eval_worker(num):
    nround = 50
    model = k.models.load_model("{}/m{}.keras".format(model_path, num))
    bot = BOT(model, verbos=0)

    n_dizhu_win, n_farmer_win = 0, 0
    for _ in range(nround):
        cards = rules.CARDS.copy()
        np.random.shuffle(cards)
        
        arena = ARENA(verbos=0, cards=cards.copy())
        arena.registerbot([bot, bot_rival, bot_rival])
        arena.wholegame()
        n_dizhu_win += (arena.winner == 0)
        
        arena2 = ARENA(verbos=0, cards=cards.copy())
        arena2.registerbot([bot_rival, bot, bot])
        arena2.wholegame()
        n_farmer_win += (arena2.winner != 0)
    return num, n_dizhu_win, n_farmer_win


def model_eval():
    mp.set_start_method('spawn')

    parts = np.linspace(0, maxnum//50, n_segment, dtype=int)
    parts = (parts + 1) * 50

    f = open(log_file, 'w', buffering=1)
    with mp.Pool(8) as p:
        for i in range(len(parts)-1):
            params = np.arange(parts[i], parts[i+1], 50)
            res = p.map(model_eval_worker, params)
            res = np.array(res)
            f.write(np.array2string(res, separator=', '))
            f.write('\n')
            print(datetime.now(), res[0][0])
    f.close()
    return res

if __name__ == '__main__':
    res = model_eval()

