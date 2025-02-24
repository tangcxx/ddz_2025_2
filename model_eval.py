# 以 "model_base2/m{}.keras".format(386150) 为基准，评估不同训练方法的表现

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

# # base2: 
# nround = 50
# maxnum = 395300
# len_segment = 80
# model_path = 'model_base2'
# from bot_base2 import BOT

# # sarsa: 
# nround = 50
# maxnum = 93200
# minnum = 50
# len_segment = 80
# model_path = 'model_sarsa'
# from bot_sarsa import BOT

# # base3: 
# nround = 50
# maxnum = 115650
# minnum = 19850
# len_segment = 80
# model_path = 'model_base3'
# from bot_base3 import BOT

# base4: 
nround = 50
maxnum = 181850
minnum = 96850
len_segment = 80
model_path = 'model_base4'
from bot_base4 import BOT

# # aug: 
# nround = 50
# maxnum = 198750
# minnum = 94250
# len_segment = 80
# model_path = 'model_aug'
# from bot_aug import BOT

def model_eval_worker(num):
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
        
        total_win = n_dizhu_win + n_farmer_win
        win_percent = total_win / (nround * 2)
    return num, n_dizhu_win, n_farmer_win, total_win, win_percent


def model_eval():
    mp.set_start_method('spawn')

    nums = np.arange(minnum, maxnum + 50, 50)
    nums_segs = [nums[i:i + len_segment] for i in range(0, len(nums), len_segment)]

    f = open('{}/eval.txt'.format(model_path), 'a', buffering=1)
    with mp.Pool(8) as p:
        for nums_seg in nums_segs:
            res = p.map(model_eval_worker, nums_seg)
            for r in res:
                f.write("{}\t{}\t{}\t{}\t{}\n".format(*r))
            print(datetime.now(), res[-1, 0])
    f.close()
    return res

if __name__ == '__main__':
    res = model_eval()

