from scipy.signal import butter, lfilter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

def butterBandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band', output="ba")

def butterBandpassFilter(data, lowcut, highcut, fs, order=3):
    b, a = butterBandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, padlen=0)
    return y

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



if __name__ == '__main__':
    signal = np.random.random((1000, ))

    filtered_signal = butteBandpassFilter(signal, 25, 30, 100 , 1)

    plt.plot(signal)
    plt.plot(filtered_signal)
    plt.show()