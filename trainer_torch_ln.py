# 基于trainer_three_torch, 模型中的BatchNorm2D改为LayerNorm

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
nround_eval = 200
bots_rival = [bot_douzero.BOT(), bot_douzero.BOT(), bot_douzero.BOT()]

# 训练参数
from arena import ARENA
from bot_torch_ln import BOT, Model

modelpath = "model_torch_ln"
iterstart=30500
model_freq = 100

nproc = 6
nmatch_per_iter = 24
batch_size = 32
epsilonstep=0.999
epsilonmin=0.01

bce = nn.BCELoss()
# loss = nn.BCEWithLogitsLoss

model_subs = [Model(), Model(), Model()]

def selfplay(args):
    ws_0, ws_1, ws_2, epsilon = args
    wss = [ws_0, ws_1, ws_2]
    bots = [[], [], []]
    for pos in range(3):
        model_subs[pos].load_state_dict(wss[pos])
        bots[pos] = BOT(models=model_subs, epsilon=epsilon)
    arena = ARENA(RECORD=False)
    arena.registerbot(bots)
    arena.wholegame()
    y = 1 if arena.winner == 0 else 0
    
    xs = [[], [], []]
    ys = [[], [], []]
    for pos in range(3):
        if bots[pos].xs:
            xs[pos] = np.concatenate(bots[pos].xs)
            ys[pos] = np.zeros(len(bots[pos].xs)) + y

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
        epsilon = max(epsilonstep ** iter, epsilonmin)
        res = p.map(selfplay, [(models[0].state_dict(), models[1].state_dict(), models[2].state_dict(), epsilon)] * nmatch_per_iter)
        xss, yss = [[], [], []], [[], [], []]
        xss = [[r[0][pos] for r in res if len(r[0][pos])>0] for pos in range(3)]
        yss = [[r[1][pos] for r in res if len(r[0][pos])>0] for pos in range(3)]

        xss = [np.concatenate(xss[pos]) for pos in range(3)]
        yss = [np.concatenate(yss[pos]) for pos in range(3)]

        for xs, ys, model, optimizer in zip(xss, yss, models, optimizers):
            model.train()
            if len(ys) == 0:
                continue
            xs, ys = torch.from_numpy(xs).float(), torch.from_numpy(ys[:, np.newaxis]).float()
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

        print(datetime.now(), iter)
        f_log.write(f"{datetime.now()}\t{iter}\n")
        iter += 1
        if iter % model_freq == 0:
            checkpoint_save(iter, models, optimizers)

            wins = p.map(eval, [(models[0].state_dict(), models[1].state_dict(), models[2].state_dict())] * nround_eval)
            wins = np.array(wins).sum(axis=0)
            wins_total = wins.sum()
            f_eval.write(f"{iter}\t{wins[0]}\t{wins[1]}\t{wins_total}\t{wins_total/(nround_eval*2)}\n")

        if iter % 200 == 0:
            p.close()
            p.join()
            p = mp.Pool(nproc)



#%%
if __name__ == '__main__':
   train()

