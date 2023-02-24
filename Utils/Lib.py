import numpy as np
from scipy.ndimage import label


def movingAverage(x, n=5):
    def moveAverage(a):
        ret = np.cumsum(a.filled(0))
        ret[n:] = ret[n:] - ret[:-n]
        counts = np.cumsum(~a.mask)
        counts[n:] = counts[n:] - counts[:-n]
        ret[~a.mask] /= counts[~a.mask]
        ret[a.mask] = np.nan
        return ret

    mx = np.ma.masked_array(x, np.isnan(x))
    x = moveAverage(mx)

    return x


def interPolateDistance(x):
    mask = np.isnan(x)
    if np.sum(mask) > 0:
        # f = interp1d(np.flatnonzero(~mask), np.flatnonzero(~mask), kind="linear")
        # x[mask] = f(np.flatnonzero(mask))
        x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])
    return x
