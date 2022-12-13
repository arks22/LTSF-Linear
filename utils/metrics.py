import numpy as np


def RSE(pd, gt):
    return np.sqrt(np.sum((gt - pd) ** 2)) / np.sqrt(np.sum((gt - gt.mean()) ** 2))


def CORR(pd, gt):
    u = ((gt - gt.mean(0)) * (pd - pd.mean(0))).sum(0)
    d = np.sqrt(((gt - gt.mean(0)) ** 2 * (pd - pd.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pd, gt):
    return np.mean(np.abs(pd - gt))


def MSE(pd, gt):
    return np.mean((pd - gt) ** 2)


def RMSE(pd, gt):
    return np.sqrt(MSE(pd, gt))


def MAPE(pd, gt):
    return np.mean(np.abs((pd - gt) / gt))


def MSPE(pd, gt):
    return np.mean(np.square((pd - gt) / gt))

def R_LAST(pd, gt):
    pd_last = pd[:, -1]
    gt_last = gt[:, -1]
    r = np.corrcoef(pd_last, gt_last)[0,1]
    return r

def R_FIRST(pd, gt):
    pd = pd[:, 0]
    gt = gt[:, 0]
    r = np.corrcoef(pd, gt)[0,1]
    return r


def metrics(pd, gt):
    mae = MAE(pd, gt)
    mse = MSE(pd, gt)
    rmse = RMSE(pd, gt)
    mape = MAPE(pd, gt)
    mspe = MSPE(pd, gt)
    rse = RSE(pd, gt)
    corr = CORR(pd, gt)
    r_first = R_FIRST(pd, gt)
    r_last = R_LAST(pd, gt)

    return mae, mse, rmse, mape, mspe, rse, corr, r_first, r_last
