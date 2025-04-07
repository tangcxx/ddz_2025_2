# 以 bot_zero 为基准，评估不同训练方法的表现
# 评估地主农民三个模型分开的训练方法

import multiprocessing as mp
from datetime import datetime

import numpy as np
import torch
import rules
import arena as arn
ARENA = arn.ARENA

import bot_douzero
bot_rivals = [bot_douzero.BOT(0), bot_douzero.BOT(0), bot_douzero.BOT(0)]

# 参数
# three: 
nproc = 6
nround = 1000
maxid = 654100
minid = 500000
len_segment = 80
model_freq = 100
model_path = 'model_torch_pool'
from bot_torch_ln import BOT, Model

def model_eval_worker(id):
    torch.set_num_threads(1)
    models = [Model(), Model(), Model()]
    cp = torch.load(f"{model_path}/cp{id}.pt")
    for pos in range(3):
        models[pos].load_state_dict(cp["models_state_dict"][pos])
        models[pos].eval()
    bots = [BOT(models, verbos=0), BOT(models, verbos=0), BOT(models, verbos=0)]

    n_dizhu_win, n_farmer_win = 0, 0
    for _ in range(nround):
        cards = rules.CARDS.copy()
        np.random.shuffle(cards)
        
        arena = ARENA(verbos=0, cards=cards.copy())
        arena.registerbot([bots[0], bot_rivals[1], bot_rivals[2]])
        arena.wholegame()
        n_dizhu_win += (arena.winner == 0)
        
        arena2 = ARENA(verbos=0, cards=cards.copy())
        arena2.registerbot([bot_rivals[0], bots[1], bots[2]])
        arena2.wholegame()
        n_farmer_win += (arena2.winner != 0)
        
        total_win = n_dizhu_win + n_farmer_win
        win_percent = total_win / (nround * 2)
    return id, n_dizhu_win, n_farmer_win, total_win, win_percent

def model_eval(minid, maxid):
    mp.set_start_method('spawn')

    ids = np.arange(minid, maxid + model_freq, model_freq)
    ids_segs = [ids[i:i + len_segment] for i in range(0, len(ids), len_segment)]

    f = open('{}/eval.txt'.format(model_path), 'a', buffering=1)
    with mp.Pool(nproc) as p:
        for ids_seg in ids_segs:
            res = p.map(model_eval_worker, ids_seg)
            for r in res:
                f.write("{}\t{}\t{}\t{}\t{}\n".format(*r))
            print(datetime.now(), res[-1][0])
    f.close()
    return res

if __name__ == '__main__':
    res = model_eval(minid, maxid)

