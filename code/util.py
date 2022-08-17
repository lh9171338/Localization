import numpy as np


def calc_error(pred, gt):
    error = np.linalg.norm(pred - gt, axis=-1)
    return error


def calc_rmse(pred, gt):
    error = calc_error(pred, gt)
    rmse = np.sqrt((error ** 2).mean(axis=-1))
    return rmse
