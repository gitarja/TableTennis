import numpy as np
from scipy.ndimage import label
from scipy.interpolate import interp1d
from numpy import array
from scipy.signal import savgol_filter

from scipy.signal import wiener
import transforms3d
from scipy.spatial.transform import Rotation as R
import os
from scipy.signal import butter, lfilter


def computePowerSpectrum(x, fs=100):
    dt = 1 / fs
    T = len(x) * dt
    xf = np.fft.fft(x - x.mean())  # Compute Fourier transform of x
    Sxx = 2 * dt ** 2 / T * (xf * xf.conj())  # Compute spectrum
    Sxx = Sxx[:int(len(x) / 2)+1]  # Ignore negative frequencies
    p = Sxx.real

    f = np.fft.rfftfreq(len(x), d=1. / fs)

    return f, p
def relativeDifference(v1, v2):
    dem = np.max(np.vstack([np.abs(v1), np.abs(v2)]).T, axis=-1)
    dist = np.abs(v1 - v2) / dem
    return dist

def percentageChange(v1, v2):
    dist = (v2 - v1) / np.abs(v1)

    return dist

def fit180(deg):

    x = deg % 360


    x[x>180] = x[x>180] - 360
    x[x < -180] = x[x < -180] + 360

    return x


def cartesianToSpher(vector, swap=False):
    xyz = np.copy(vector)
    if swap:

        R_m = np.array([np.squeeze(R.from_euler("zx", [180, 270], degrees=True).as_matrix()) for i in range(len(xyz))])
        xyz = np.squeeze(np.matmul(R_m, np.expand_dims(xyz, 2)))


    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    r = np.sqrt(xy + xyz[:, 2] ** 2)
    # elv = np.arctan2(np.sqrt(xy), xyz[:, 2])*180/ np.pi   # for elevation angle defined from Z-axis down
    elv = np.rad2deg(np.arctan2(xyz[:, 2], np.sqrt(xy))) # for elevation angle defined from XY-plane up
    az = np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0])) - 90

    az = fit180(az)

    return r, az , elv

def butterLowPass(lowcut, fs, order=5):
    return butter(order, lowcut, fs=fs, btype='low')

def butterLowPassFilter(data, lowcut, fs, order=5):
    b, a = butterLowPass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

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