# 基于lr + scaledloss2, 地主农民分三个模型

#%%
import keras as k
import tensorflow as tf
import numpy as np 
import multiprocessing as mp
from datetime import datetime

from arena import ARENA
from bot_three import BOT

modelpath = "model_three_3"
iterstart=9100

nproc = 8
nmatch_per_iter = 24
batch_size = 32
epsilonstep=0.999
epsilonmin=0.01

bce = k.losses.binary_crossentropy
model_subs = [BOT.createmodel(), BOT.createmodel(), BOT.createmodel()]

def selfplay(args):
    ws_0, ws_1, ws_2, epsilon = args
    wss = [ws_0, ws_1, ws_2]
    bots = [[], [], []]
    for pos in range(3):
        model_subs[pos].set_weights(wss[pos])
        bots[pos] = BOT(model=model_subs[pos], epsilon=epsilon)
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

def train():
    mp.set_start_method('spawn')
    p = mp.Pool(nproc)
    iter = iterstart
    lossL = np.zeros(200) - np.log(0.5)
    models = [k.models.load_model("{0}/{1}/m{2}.keras".format(modelpath, pos, iter)) for pos in range(3)]
    
    f = open("{}/log.txt".format(modelpath), "a", buffering=1)
    while True:
        epsilon = max(epsilonstep ** iter, epsilonmin)
        res = p.map(selfplay, [(models[0].get_weights(), models[1].get_weights(), models[2].get_weights(), epsilon)] * nmatch_per_iter)
        xss, yss = [[], [], []], [[], [], []]
        xss = [[r[0][pos] for r in res if len(r[0][pos])>0] for pos in range(3)]
        yss = [[r[1][pos] for r in res if len(r[0][pos])>0] for pos in range(3)]

        xss = [np.concatenate(xss[pos]) for pos in range(3)]
        yss = [np.concatenate(yss[pos]) for pos in range(3)]

        loss1 = np.mean([bce(yss[pos], models[pos](xss[pos])[:,0]).numpy() for pos in range(3)])
        lossL[(iter - iterstart) % len(lossL)] = loss1

        for xs, ys, model in zip(xss, yss, models):
            if len(ys) == 0:
                continue
            indices = list(range(ys.shape[0]))
            np.random.shuffle(indices)
            indices_lists = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
            for indices in indices_lists:
                xs_batch = xs[indices]
                ys_batch = ys[indices]
                loss_factor = (len(xs_batch) / batch_size) ** 0.5
                with tf.GradientTape() as tape:
                    loss = bce(ys_batch, model(xs_batch, training=True)[:,0])
                    loss = tf.reduce_mean(loss) * loss_factor
                grads = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_mean = np.round(np.mean(lossL[0:(iter - iterstart + 1)]), 3)
        print(datetime.now(), iter, 
              loss_mean, 
              np.round(loss1, 3))
        f.write("{0} {1} {2} {3}\n".format(datetime.now(), iter, loss_mean, np.round(loss1, 3)))
        iter += 1
        if iter % 50 == 0:
            for pos, model in enumerate(models):
                model.save("{0}/{1}/m{2}.keras".format(modelpath, pos, iter))

#%%
if __name__ == '__main__':
   train()

