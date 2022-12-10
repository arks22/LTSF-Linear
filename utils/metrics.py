import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def R_LAST(pred, true):
    pred_last = pred[:, -1, 0]
    true_last = true[:, -1, 0]
    r = np.corrcoef(pred_last, true_last)[0,1]
    return r

def R_FIRST(pred, true):
    pred = pred[:, 0, 0]
    true = true[:, 0, 0]
    r = np.corrcoef(pred, true)[0,1]
    return r


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    r_first = R_FIRST(pred, true)
    r_last = R_LAST(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr, r_last
