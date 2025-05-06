# 基于fairtrainer_aug2
# 使用不作弊模型生成数据，用作弊模型的估值作为奖励信号

#%%
import os
import torch
from torch import nn
import numpy as np 
import multiprocessing as mp
from datetime import datetime

## 模型评价
import bot_douzero
from rules import CARDS
nround_eval = 1000
bots_rival = [bot_douzero.BOT(), bot_douzero.BOT(), bot_douzero.BOT()]

# 训练参数
from arena import ARENA
from fairbot_sl_value import BOT, Model

modelpath = "fairmodel_sl_value"
iterstart=10
model_freq = 5

nproc = 6
nmatch_per_iter = 128
batch_size = 128
nround_pool_recycle = 50

bce = nn.BCELoss()
# loss = nn.BCEWithLogitsLoss

model_subs = [Model(), Model(), Model()]

import bot_torch_ln
cp_cheater = torch.load("model_tree2/cp14315.pt")
models_cheater = []
for pos in range(3):
    model = bot_torch_ln.Model()
    model.eval()
    model.load_state_dict(cp_cheater["models_state_dict"][pos])
    models_cheater.append(model)

def selfplay(args):
    torch.set_num_threads(1)
    ws_0, ws_1, ws_2 = args
    wss = [ws_0, ws_1, ws_2]
    for pos in range(3):
        model_subs[pos].load_state_dict(wss[pos])

    arena = ARENA(RECORD=True)
    arena.registerbot([BOT(models=model_subs), BOT(models=model_subs), BOT(models=model_subs)])
    arena.wholegame()
    y = 1 if arena.winner == 0 else 0

    zs_branch = [bot.zs for bot in arena.bot]
    xs_branch = [bot.xs for bot in arena.bot]
    xs_cheating_branch = [bot.xs_cheating for bot in arena.bot]
    
    zs_tree = [zs_branch]
    xs_tree = [xs_branch]
    xs_cheating_tree = [xs_cheating_branch]

    for a in arena.records[0:-1]:
        pos = a.pos
        choices = a.getChoices()
        if len(choices) <= 1:
            continue
        a1 = a.copy()
        a1.registerbot([BOT(models=model_subs), BOT(models=model_subs), BOT(models=model_subs)])
        zs, xs, preds = a1.bot[pos].get_dizhu_win_probs(a1, choices)
        scores = preds if pos == 0 else 1 - preds
        for idx in np.argsort(scores)[-5:-1]:
            a2 = a.copy()
            a2.registerbot([BOT(models=model_subs), BOT(models=model_subs), BOT(models=model_subs)])
            bot_pos = a2.bot[pos]
            bot_pos.zs.append(zs[idx:(idx+1)])
            bot_pos.xs.append(xs[idx:(idx+1)])
            bot_pos.xs_cheating.append(bot_pos.getdata_cheating(a2, choices[idx:idx+1]))
            a2.play(choices[idx])
            a2.wholegame()
            zs_branch = [bot.zs for bot in a2.bot]
            xs_branch = [bot.xs for bot in a2.bot]
            xs_cheating_branch = [bot.xs_cheating for bot in a2.bot]
    
            zs_tree.append(zs_branch)
            xs_tree.append(xs_branch)
            xs_cheating_tree.append(xs_cheating_branch)
            
    zs = [[], [], []]
    xs = [[], [], []]
    xs_cheating = [[], [], []]
    for zs_branch, xs_branch, xs_cheating_branch in zip(zs_tree, xs_tree, xs_cheating_tree):
        for pos in range(3):
            zs[pos].extend(zs_branch[pos])
            xs[pos].extend(xs_branch[pos])
            xs_cheating[pos].extend(xs_cheating_branch[pos])
    for pos in range(3):
        zs[pos] = torch.concatenate(zs[pos])
        xs[pos] = torch.concatenate(xs[pos])
        xs_cheating[pos] = torch.concatenate(xs_cheating[pos])
    return zs, xs, xs_cheating


def eval(args):
    for pos in range(3):
        model_subs[pos].load_state_dict(args[pos])
        bots = [BOT(models=model_subs), BOT(models=model_subs), BOT(models=model_subs)]

    n_dizhu_win, n_farmer_win = 0, 0

    cards = CARDS.copy()
    np.random.shuffle(cards)
    
    arena = ARENA(verbos=0, cards=cards.copy())
    arena.registerbot([bots[0], bots_rival[1], bots_rival[2]])
    arena.wholegame()
    n_dizhu_win += (arena.winner == 0)
    
    arena2 = ARENA(verbos=0, cards=cards.copy())
    arena2.registerbot([bots_rival[0], bots[1], bots[2]])
    arena2.wholegame()
    n_farmer_win += (arena2.winner != 0)

    return n_dizhu_win, n_farmer_win

def checkpoint_save(iter, models, optizimers):
    torch.save(
                    {
                        "models_state_dict": (models[0].state_dict(), models[1].state_dict(), models[2].state_dict()),
                        "optizimers_state_dict": (optizimers[0].state_dict(), optizimers[1].state_dict(), optizimers[2].state_dict())
                    },
                    f"{modelpath}/cp{iter}.pt"
                )

def checkpoint_load(iter, models, optimizers):
    cp = torch.load(f"{modelpath}/cp{iter}.pt")
    for pos in range(3):
        models[pos].load_state_dict(cp["models_state_dict"][pos])
        optimizers[pos].load_state_dict(cp["optizimers_state_dict"][pos])

def train():
    mp.set_start_method('spawn')
    p = mp.Pool(nproc)
    iter = iterstart
    models = [Model(), Model(), Model()]
    optimizers = [torch.optim.Adam(models[pos].parameters(), lr=1e-4) for pos in range(3)]
    if iter == 0:
        os.makedirs(f"{modelpath}", exist_ok=True)
        checkpoint_save(iter, models, optimizers)
    else:
        checkpoint_load(iter, models, optimizers)
    
    f_log = open("{}/log.txt".format(modelpath), "a", buffering=1)
    f_eval = open("{}/eval.txt".format(modelpath), "a", buffering=1)
    while True:
        res = p.map(selfplay, [(models[0].state_dict(), models[1].state_dict(), models[2].state_dict())] * nmatch_per_iter)

        zss, xss, xss_cheating = [[], [], []], [[], [], []], [[], [], []]
        for zs, xs, xs_cheating in res:
            for pos in range(3):
                zss[pos].append(zs[pos])
                xss[pos].append(xs[pos])
                xss_cheating[pos].append(xs_cheating[pos])
                

        for zs, xs, xs_cheating, model, optimizer, model_cheater in zip(zss, xss, xss_cheating, models, optimizers, models_cheater):
            model.train()
            if len(xs) == 0:
                continue
            zs = torch.concatenate(zs).float()
            xs = torch.concatenate(xs).float()
            xs_cheating = torch.concatenate(xs_cheating).float()
            with torch.no_grad():
                # ys = model_cheater(xs_cheating).numpy()[:,0]
                ys = model_cheater(xs_cheating)
                
            indices = list(range(xs.shape[0]))
            np.random.shuffle(indices)
            indices_lists = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
            for indices in indices_lists:
                optimizer.zero_grad()
                zs_batch = zs[indices]
                xs_batch = xs[indices]
                ys_batch = ys[indices]
                loss_factor = (len(xs_batch) / batch_size) ** 0.5
                loss = bce(model(zs_batch, xs_batch), ys_batch) * loss_factor 
                loss.backward()
                optimizer.step()

        iter += 1
        if iter % model_freq == 0:
            checkpoint_save(iter, models, optimizers)

            wins = p.map(eval, [(models[0].state_dict(), models[1].state_dict(), models[2].state_dict())] * nround_eval)
            wins = np.array(wins).sum(axis=0)
            wins_total = wins.sum()
            f_eval.write(f"{iter}\t{wins[0]}\t{wins[1]}\t{wins_total}\t{wins_total/(nround_eval*2)}\n")
        print(datetime.now(), iter)
        f_log.write(f"{datetime.now()}\t{iter}\n")

        if iter % nround_pool_recycle == 0:
            p.close()
            p.join()
            p = mp.Pool(nproc)

#%%
if __name__ == '__main__':
   train()
