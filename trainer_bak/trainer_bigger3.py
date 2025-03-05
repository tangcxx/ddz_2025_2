#训练了1小时，效果不如bigger

#%%
import keras as k
import numpy as np 
import multiprocessing as mp
from datetime import datetime

class PARAM:
    def __init__(self, modelpath, ARENA, BOT, iterstart=0, nproc=8, epsilonstep=0.999, epsilonmin=0.01, learning_rate=None, batch_size=32):
        self.BOT = BOT
        self.ARENA = ARENA
        self.iterstart = iterstart
        self.nproc = nproc
        self.epsilonstep = epsilonstep
        self.epsilonmin = epsilonmin
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.modelpath = modelpath
        # self.weightpath = "{0}/weights".format(modelpath)
        self.model_sub = self.BOT.createmodel()

from arena import ARENA
from bot_bigger3 import BOT
param = PARAM("model_bigger3", ARENA, BOT, iterstart=0)

def selfplay(args):
    ws, epsilon = args
    param.model_sub.set_weights(ws)
    bot = param.BOT(model=param.model_sub, epsilon=epsilon)
    arena = param.ARENA(RECORD=False)
    arena.registerbot([bot, bot, bot])
    arena.wholegame()
    y = 1 if arena.winner == 0 else 0
    
    xs = np.concatenate(bot.xs)
    ys = np.zeros(len(bot.xs)) + y
    return [ys, xs]

def train():
    mp.set_start_method('spawn')
    p = mp.Pool(param.nproc)
    bce = k.losses.binary_crossentropy
    iter = param.iterstart
    lossL = np.zeros(200) - np.log(0.5)
    model = k.models.load_model("{0}/m{1}.keras".format(param.modelpath, iter))
    if param.learning_rate:
        model.compile(loss="binary_crossentropy",
                      optimizer=k.optimizers.Adam(learning_rate=param.learning_rate))
    f = open("{}/log.txt".format(param.modelpath), "a", buffering=1)
    while True:
        epsilon = max(param.epsilonstep ** iter, param.epsilonmin)
        res = p.map(selfplay, [(model.get_weights(), epsilon)] * 8)
        ys = np.concatenate([r[0] for r in res])
        xs = np.concatenate([r[1] for r in res])

        loss1 = bce(ys, model(xs)[:,0]).numpy()
        lossL[iter % len(lossL)] = loss1

        model.fit(xs, ys,
                  batch_size=param.batch_size,
                  epochs=1,
                  verbose=0)
        print(datetime.now(), iter, 
              np.round(np.mean(lossL), 3), 
              np.round(np.mean(loss1), 3))
        f.write("{0} {1} {2} {3}\n".format(datetime.now(), iter, np.round(np.mean(lossL), 3), np.round(np.mean(loss1), 3)))
        iter += 1
        if iter % 50 == 0:
            model.save("{0}/m{1}.keras".format(param.modelpath, iter))

#%%
if __name__ == '__main__':
   train()

