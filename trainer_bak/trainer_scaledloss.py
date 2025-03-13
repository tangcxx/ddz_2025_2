# 基于lr, 基于batch的样本数量修改loss, loss_factor = loss_factor = (len(xs_batch) / batch_size

#%%
import keras as k
import tensorflow as tf
import numpy as np 
import multiprocessing as mp
from datetime import datetime

from arena import ARENA
from bot_scaledloss import BOT

modelpath = "model_scaledloss"
iterstart=5650

nproc = 8
nmatch_per_iter = 8
batch_size = 32
epsilonstep=0.999
epsilonmin=0.01

bce = k.losses.binary_crossentropy
model_sub = BOT.createmodel()

def selfplay(args):
    ws, epsilon = args
    model_sub.set_weights(ws)
    bot = BOT(model=model_sub, epsilon=epsilon)
    arena = ARENA(RECORD=False)
    arena.registerbot([bot, bot, bot])
    arena.wholegame()
    y = 1 if arena.winner == 0 else 0
    
    xs = np.concatenate(bot.xs)
    ys = np.zeros(len(bot.xs)) + y
    return [ys, xs]

def train():
    mp.set_start_method('spawn')
    p = mp.Pool(nproc)
    iter = iterstart
    lossL = np.zeros(200) - np.log(0.5)
    model = k.models.load_model("{0}/m{1}.keras".format(modelpath, iter))
    f = open("{}/log.txt".format(modelpath), "a", buffering=1)
    while True:
        epsilon = max(epsilonstep ** iter, epsilonmin)
        res = p.map(selfplay, [(model.get_weights(), epsilon)] * nmatch_per_iter)
        ys = np.concatenate([r[0] for r in res])
        xs = np.concatenate([r[1] for r in res])

        loss1 = bce(ys, model(xs)[:,0]).numpy()
        lossL[(iter - iterstart) % len(lossL)] = loss1

        indices = list(range(ys.shape[0]))
        np.random.shuffle(indices)
        indices_lists = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

        for indices in indices_lists:
            xs_batch = xs[indices]
            ys_batch = ys[indices]
            n_batch = len(xs_batch)
            loss_factor = len(xs_batch) / batch_size
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
            model.save("{0}/m{1}.keras".format(modelpath, iter))

#%%
if __name__ == '__main__':
   train()

