import keras as k
import numpy as np 
import multiprocessing as mp
from datetime import datetime

class PARAM:
    def __init__(self, modelpath, ARENA, BOT, iterstart=135, nproc=8, epsilonstep=0.999, epsilonmin=0.01, learning_rate=None, batch_size=32):
        self.BOT = BOT
        self.ARENA = ARENA
        self.iterstart = iterstart
        self.nproc = nproc
        self.epsilonstep = epsilonstep
        self.epsilonmin = epsilonmin
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.modelpath = modelpath
        self.weightpath = "{0}/weights".format(modelpath)
        self.model_sub = self.BOT.createmodel()

from bot_nn import BOT
from arena import ARENA
param = PARAM("weights", ARENA, BOT)

def selfplay(args):
    ws, epsilon = args
    param.model_sub.set_weights(ws)
    bot = param.BOT(model=param.model_sub, epsilon=epsilon)
    arena = param.ARENA(RECORD=True)
    arena.registerbot([bot, bot, bot])
    arena.wholegame()
    nmoves = len(arena.records)
    idx = np.random.randint(nmoves)
    data = param.BOT.getdata(arena.records[idx])
    y = 1 if arena.winner == 0 else 0
    return [y, data]

def train():
    mp.set_start_method('spawn')
    p = mp.Pool(param.nproc)
    bce = k.losses.binary_crossentropy
    iter = param.iterstart
    lastsave = iter
    lossL = []
    model = k.models.load_model("{0}/m{1}.keras".format(param.modelpath, iter))
    if param.learning_rate:
        model.compile(loss="binary_crossentropy",
                      optimizer=k.optimizers.Adam(learning_rate=param.learning_rate))
    while True:
        epsilon = max(param.epsilonstep ** iter, param.epsilonmin)
        res = p.map(selfplay, [(model.get_weights(), epsilon)] * param.batch_size)
        y = np.array([r[0] for r in res])
        cards = np.concatenate([r[1] for r in res])

        loss1 = bce(y, model.predict(cards)[:,0], verbose=0).numpy()
        lossL.append(loss1)
        thres = np.mean(lossL) - 1.65 * np.std(lossL)
        if loss1 < thres or iter - lastsave == 50:
            model.save("{0}/m{1}.keras".format(param.modelpath, iter))
            model.save_weights("{0}/w{1}.weights.h5".format(param.weightpath, iter))
            lastsave = iter
        if len(lossL) == 200:
            lossL = lossL[1:]

        model.fit(cards, y,
                  batch_size=param.batch_size,
                  epochs=1,
                  verbose=0)
        loss2 = bce(y, model.predict(cards)[:,0], verbose=0).numpy()
        print(datetime.now(), iter, np.round(thres, 3), np.round(np.mean(lossL), 3), np.array([loss1, loss2]).round(3))
        iter += 1

if __name__ == '__main__':
   train()

