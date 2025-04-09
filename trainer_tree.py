# 基于trainer_torch_ln
# 根据模型打牌，生成一个局面信息的序列
# 在序列的每一步，选择所有其他可能的选项，然后再按模型打牌，生成序列
# 一个局面信息的树，每一层有若干个子结点，对应当前局面下的不同出牌，但只有一个子结点（最高胜率的出牌）有分支，其他的都是一条线直到叶结点
# 希望具有的优点：1, 一手牌提供更多的训练数据。2, 同一个局面下的不同出牌的奖励对比能提供更有效的训练信号

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
nround_eval = 1200
bots_rival = [bot_douzero.BOT(), bot_douzero.BOT(), bot_douzero.BOT()]

# 训练参数
from arena import ARENA
from bot_torch_ln import BOT, Model

modelpath = "model_tree"
iterstart=91
model_freq = 1

nproc = 6
nmatch_per_iter = 6
batch_size = 32
nround_pool_recycle = 50

bce = nn.BCELoss()
# loss = nn.BCEWithLogitsLoss

model_subs = [Model(), Model(), Model()]

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

    xs_branch = [bot.xs for bot in arena.bot]
    ys_branch = [[y] * len(bot.xs) for bot in arena.bot]
    
    xs_tree = [xs_branch]
    ys_tree = [ys_branch]

    for a in arena.records[0:-1]:
        pos = a.pos
        choices = a.getChoices()
        if len(choices) <= 1:
            continue
        a1 = a.copy()
        a1.registerbot([BOT(models=model_subs), BOT(models=model_subs), BOT(models=model_subs)])
        choices = a1.getChoices()
        xs, preds = a1.bot[pos].get_dizhu_win_probs(a1, choices)
        scores = preds if pos == 0 else 1 - preds
        indices = np.argsort(scores)[0:-1]
        for idx in indices:
            a2 = a.copy()
            a2.registerbot([BOT(models=model_subs), BOT(models=model_subs), BOT(models=model_subs)])
            bot_pos = a2.bot[pos]
            bot_pos.xs.append(bot_pos.getdata(arena, choices[idx:idx+1]))
            a2.play(choices[idx])
            a2.wholegame()
            y = 1 if a2.winner == 0 else 0
            xs_branch = [bot.xs for bot in a2.bot]
            ys_branch = [[y] * len(bot.xs) for bot in a2.bot]
            xs_tree.append(xs_branch)
            ys_tree.append(ys_branch)
            
    xs = [[], [], []]
    ys = [[], [], []]
    for xs_branch, ys_branch in zip(xs_tree, ys_tree):
        for pos in range(3):
            xs[pos].extend(xs_branch[pos])
            ys[pos].extend(ys_branch[pos])
    for pos in range(3):
        xs[pos] = torch.concatenate(xs[pos])
    return xs, ys


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

        xss, yss = [[], [], []], [[], [], []]
        for xs, ys in res:
            for pos in range(3):
                xss[pos].append(xs[pos])
                yss[pos].extend(ys[pos])

        for xs, ys, model, optimizer in zip(xss, yss, models, optimizers):
            model.train()
            if len(ys) == 0:
                continue
            xs = torch.concatenate(xs).float()
            ys = torch.tensor(ys).reshape(-1,1).float()
            indices = list(range(ys.shape[0]))
            np.random.shuffle(indices)
            indices_lists = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
            for indices in indices_lists:
                optimizer.zero_grad()
                xs_batch = xs[indices]
                ys_batch = ys[indices]
                loss_factor = (len(xs_batch) / batch_size) ** 0.5
                loss = bce(model(xs_batch), ys_batch) * loss_factor 
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
