import numpy as np
from scipy.ndimage import label
from scipy.interpolate import interp1d
from numpy import array
from scipy.signal import savgol_filter

from scipy.signal import wiener
import transforms3d

import os


def checkMakeDir(path):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)


def wienerFilter(x, n=3):

    return wiener(x, n)

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


def savgolFilter(x, n=5):
    x = savgol_filter(x, n, 1)

    return x



def interPolateDistance(x):
    mask = np.isnan(x)
    if np.sum(mask) > 0:
        # f = interp1d(np.flatnonzero(~mask), np.flatnonzero(~mask), kind="linear")
        # x[mask] = f(np.flatnonzero(mask))
        x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])
    return x



def interExtrapolate(x):

    def extrap1d(interpolator):
        xs = interpolator.x
        ys = interpolator.y

        def pointwise(x):
            if x < xs[0]:
                return ys[0]
            elif x > xs[-1]:
                return ys[-1]
            else:
                return interpolator(x)

        def ufunclike(xs):
            return array(list(map(pointwise, array(xs))))

        return ufunclike

    x_train = np.copy(x)
    mask = np.isnan(x)
    f = interp1d(np.flatnonzero(~mask), x_train[~mask], bounds_error=False, kind="linear", fill_value="extrapolate")
    f_x = extrap1d(f)

    # fillin the nan values
    x[mask] = f_x(np.flatnonzero(mask))


    return x


def interExtrapolate2(x):

    def extrap1d(interpolator):
        xs = interpolator.x
        ys = interpolator.y

        def pointwise(x):
            if x < xs[0]:
                return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
            elif x > xs[-1]:
                return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            else:
                return interpolator(x)

        def ufunclike(xs):
            return array(list(map(pointwise, array(xs))))

        return ufunclike

    x_train = np.copy(x)
    mask = np.isnan(x)
    f = interp1d(np.flatnonzero(~mask), x_train[~mask], bounds_error=False, kind="linear", fill_value="extrapolate")
    f_x = extrap1d(f)

    # fillin the nan values
    x[mask] = f_x(np.flatnonzero(mask))


    return x