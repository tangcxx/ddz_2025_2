# 以 "model_base2/m{}.keras".format(386150) 为基准，评估不同训练方法的表现
# 评估地主农民三个模型分开的训练方法

import multiprocessing as mp
from datetime import datetime

import numpy as np
import keras as k
import rules
import arena as arn
import bot_base2
ARENA = arn.ARENA

model_rival = k.models.load_model("e:/ddz_2025_2_model/model_base2/m{}.keras".format(386150))
bot_rival = bot_base2.BOT(model_rival, verbos=0)

# 参数
# three: 
nround = 50
maxnum = 76200
minnum = 75250
len_segment = 80
model_freq = 50
model_path = 'model_three_3'
from bot_three import BOT

def model_eval_worker(num):
    models = [k.models.load_model("{}/{}/m{}.keras".format(model_path, pos, num)) for pos in range(3)]
    bots = [BOT(models, verbos=0), BOT(models, verbos=0), BOT(models, verbos=0)]

    n_dizhu_win, n_farmer_win = 0, 0
    for _ in range(nround):
        cards = rules.CARDS.copy()
        np.random.shuffle(cards)
        
        arena = ARENA(verbos=0, cards=cards.copy())
        arena.registerbot([bots[0], bot_rival, bot_rival])
        arena.wholegame()
        n_dizhu_win += (arena.winner == 0)
        
        arena2 = ARENA(verbos=0, cards=cards.copy())
        arena2.registerbot([bot_rival, bots[1], bots[2]])
        arena2.wholegame()
        n_farmer_win += (arena2.winner != 0)
        
        total_win = n_dizhu_win + n_farmer_win
        win_percent = total_win / (nround * 2)
    return num, n_dizhu_win, n_farmer_win, total_win, win_percent

def model_eval():
    mp.set_start_method('spawn')

    nums = np.arange(minnum, maxnum + model_freq, model_freq)
    nums_segs = [nums[i:i + len_segment] for i in range(0, len(nums), len_segment)]

    f = open('{}/eval.txt'.format(model_path), 'a', buffering=1)
    with mp.Pool(8) as p:
        for nums_seg in nums_segs:
            res = p.map(model_eval_worker, nums_seg)
            for r in res:
                f.write("{}\t{}\t{}\t{}\t{}\n".format(*r))
            print(datetime.now(), res[-1][0])
    f.close()
    return res

if __name__ == '__main__':
    res = model_eval()

