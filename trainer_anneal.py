#使用所有样本
#%%
import keras as k
import numpy as np 
import multiprocessing as mp
from datetime import datetime

class PARAM:
    def __init__(self, modelpath, ARENA, BOT, iterstart=0, nproc=8, temp_step=0.99, temp_min=0.012, learning_rate=None, batch_size=32):
        self.BOT = BOT
        self.ARENA = ARENA
        self.iterstart = iterstart
        self.nproc = nproc
        self.temp_step = temp_step
        self.temp_min = temp_min
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.modelpath = modelpath
        # self.weightpath = "{0}/weights".format(modelpath)
        self.model_sub = self.BOT.createmodel()

from bot_anneal import BOT
from arena import ARENA
param = PARAM("model_anneal", ARENA, BOT, iterstart=9105)

def selfplay(args):
    ws, temperature = args
    param.model_sub.set_weights(ws)
    bot = param.BOT(model=param.model_sub, temperature=temperature)
    arena = param.ARENA(RECORD=True)
    arena.registerbot([bot, bot, bot])
    arena.wholegame()
    y = 1 if arena.winner == 0 else 0
    
    ys, xs = [], []
    for i, record in enumerate(arena.records):
        ys.append(y)
        xs.append(param.BOT.getdata(record))
    ys = np.array(ys)
    xs = np.concatenate(xs)
    return [ys, xs]
    # nmoves = len(arena.records)
    # idx = np.random.randint(nmoves)
    # data = param.BOT.getdata(arena.records[idx])
    # return [y, data]

def train():
    mp.set_start_method('spawn')
    p = mp.Pool(param.nproc)
    bce = k.losses.binary_crossentropy
    iter = param.iterstart
    # lastsave = iter
    lossL = []
    model = k.models.load_model("{0}/m{1}.keras".format(param.modelpath, iter))
    if param.learning_rate:
        model.compile(loss="binary_crossentropy",
                      optimizer=k.optimizers.Adam(learning_rate=param.learning_rate))
    while True:
        temperature = max(param.temp_step ** iter, param.temp_min)
        res = p.map(selfplay, [(model.get_weights(), temperature)] * 8)
        ys = np.concatenate([r[0] for r in res])
        xs = np.concatenate([r[1] for r in res])

        loss1 = bce(ys, model(xs)[:,0]).numpy()
        # raise ValueError(ys, model(xs).numpy()[:,0])
        lossL.append(loss1)
        # thres = np.mean(lossL) - 1.65 * np.std(lossL)
        if len(lossL) == 200:
            lossL = lossL[1:]

        model.fit(xs, ys,
                  batch_size=param.batch_size,
                  epochs=1,
                  verbose=0)
        # loss2 = bce(y, model(xs)[:,0]).numpy()
        print(datetime.now(), iter, 
            #   np.round(thres, 3), 
              np.round(np.mean(lossL), 3), 
              np.round(np.mean(loss1), 3))
        iter += 1
        # if loss1 < thres or iter - lastsave == 50:
        if iter % 50 == 0:
            model.save("{0}/m{1}.keras".format(param.modelpath, iter))
            # model.save_weights("{0}/w{1}.weights.h5".format(param.weightpath, iter))
            # lastsave = iter

#%%
if __name__ == '__main__':
   train()

