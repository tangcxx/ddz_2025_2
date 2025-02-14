# 使用所有样本
# 和 trainer_base_allrecords.py 一样, 重新训练一次作为验证
# 目标来看，小的卷积核+更多层没用。增加探索性也没用。
# 只能试试bootstrap了
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

from bot_base2 import BOT
from arena import ARENA
param = PARAM("model_base2", ARENA, BOT, iterstart=0)

def selfplay(args):
    ws, epsilon = args
    param.model_sub.set_weights(ws)
    bot = param.BOT(model=param.model_sub, epsilon=epsilon)
    arena = param.ARENA(RECORD=True)
    arena.registerbot([bot, bot, bot])
    arena.wholegame()
    y = 1 if arena.winner == 0 else 0
    
    ys, xs = [], []
    for _, record in enumerate(arena.records):
        ys.append(y)
        xs.append(param.BOT.getdata(record))
    ys = np.array(ys)
    xs = np.concatenate(xs)
    return [ys, xs]

def train():
    mp.set_start_method('spawn')
    p = mp.Pool(param.nproc)
    bce = k.losses.binary_crossentropy
    iter = param.iterstart
    lossL = []
    model = k.models.load_model("{0}/m{1}.keras".format(param.modelpath, iter))
    if param.learning_rate:
        model.compile(loss="binary_crossentropy",
                      optimizer=k.optimizers.Adam(learning_rate=param.learning_rate))
    while True:
        epsilon = max(param.epsilonstep ** iter, param.epsilonmin)
        res = p.map(selfplay, [(model.get_weights(), epsilon)] * 8)
        ys = np.concatenate([r[0] for r in res])
        xs = np.concatenate([r[1] for r in res])

        loss1 = bce(ys, model(xs)[:,0]).numpy()
        lossL.append(loss1)
        if len(lossL) == 200:
            lossL = lossL[1:]

        model.fit(xs, ys,
                  batch_size=param.batch_size,
                  epochs=1,
                  verbose=0)
        print(datetime.now(), iter, 
              np.round(np.mean(lossL), 3), 
              np.round(np.mean(loss1), 3))
        iter += 1
        if iter % 50 == 0:
            model.save("{0}/m{1}.keras".format(param.modelpath, iter))

#%%
if __name__ == '__main__':
   train()

