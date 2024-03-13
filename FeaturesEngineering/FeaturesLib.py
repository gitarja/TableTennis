import nolds
import numpy as np
from scipy.signal import lfilter, convolve, savgol_filter
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde as kde, entropy
from scipy.signal import welch, hilbert
from scipy import signal
from dtaidistance import dtw
from tslearn import metrics
from scipy import stats
from pyinform.mutualinfo import mutual_info
from pyinform.transferentropy import transfer_entropy
from Utils.Lib import movingAverage


def whichShoulder(racket, r_wirst, l_wrist):
    r_r = np.linalg.norm(racket - r_wirst)
    l_r = np.linalg.norm(racket - l_wrist)

    if r_r < l_r:
        return True
    return False


def computeVelAccV2(v: np.array, normalize=True) -> np.array:
    '''
    :param v: vector
    :param normalize: whether to use normalized velocity to compute acceleration or not. Normalization is performed by applying Duchowski's filter (https://doi.org/10.3758/BF03195486)
    :return:
    - velocity
    - velocity norm
    - acceleration
    '''
    v1 = v[1:]
    v2 = v[:-1]
    kernel_vel = np.array([0, 1, 2, 3, 2, 1, 0])  # Duchowski's filter (https://doi.org/10.3758/BF03195486)
    velocity = computeSegmentAngles(v1, v2) * 100  # convert to deg / sec
    velocity = np.pad(velocity, (0, 1), 'symmetric')
    velocity_norm = lfilter(kernel_vel, 10, velocity)
    # window_length = 5
    # if len(velocity)<5:
    #     window_length = 3
    # velocity_norm = savgol_filter(velocity, window_length, polyorder=1)
    if normalize:
        acceler = np.diff(velocity_norm, n=1, axis=0)
    else:
        acceler = np.diff(velocity, n=1, axis=0)
    acceler = np.pad(acceler, (0, 1), 'symmetric')

    # if (len(v)) > 90:
    #     import matplotlib.pyplot as plt
    #     print(len(v))
    #     print(len(velocity_norm))
    #     print(len(acceler))
    #     plt.plot(velocity_norm)
    #     plt.plot(acceler)
    #     # plt.savefig("F:\\users\\prasetia\\projects\\Animations\\TableTennis\\acceler.eps", format='eps')
    #     plt.show()

    return velocity, velocity_norm, acceler


def computeVectorsDirection(v1: np.array, v2: np.array):
    v1_u = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)  # normalize v1
    v2_u = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)  # normalize v2
    direction = np.einsum('ij,ij->i', v1_u, v2_u)
    return direction


def computeSegmentAngles(v1: np.array, v2: np.array):
    v1_u = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)  # normalize v1
    v2_u = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)  # normalize v2
    angles = np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i', v1_u, v2_u), -1.0, 1.0)))
    return angles


def computeHistBouce(ball: np.array, episodes: np.array):
    wall_bounce = []
    table_bounce = []
    for e in episodes:
        wall_bounce.append(ball[e[2], [0, 2]])
        table_bounce.append(ball[e[3], [0, 1]])

    return np.vstack(wall_bounce), np.vstack(table_bounce)


def computeVelAcc(v, speed_only=False, acc_only=False, fps=1):
    v1 = v[:-1]
    v2 = v[1:]
    speed = np.linalg.norm(v2 - v1, axis=-1) / fps
    vel = np.nansum(np.diff(v, n=1, axis=0), axis=-1) / fps
    acc = np.diff(speed, n=1, axis=-1)
    if speed_only:
        return speed
    elif acc_only:
        return acc
    return speed, vel, acc


def computeKineticEnergy(v, n_window = 5, mass = 1, fps=1):

    vector = v[-n_window:]

    velocity_mag = np.linalg.norm(np.nanmean(np.sqrt(np.square(np.diff(vector, n=1, axis=0))), axis=0) / fps)
    force = 0.5 * mass * velocity_mag ** 2

    return force

def lyapunovExponent(v, emb_dim=10, matrix_dim=4):
    """
    Computes Lyapunov Exponent for of the NNi series
    The first LE is considered as the instantaneous dominant of LE
    Recommendations for parameter settings by Eckmann et al.:
        - long recording time improves accuracy, small tau does not
        - Use large values for emb_dim
        - Matrix_dim should be ‘somewhat larger than the expected number of positive Lyapunov exponents’
        - Min_nb = min(2 * matrix_dim, matrix_dim + 4)
    :param nni:
    :param rpeaks:
    :param emb_dim:
    :param matrix_dim:expected dimension of lyapunov exponential
    :return: the first LE
    """
    min_nb = min(2 * matrix_dim, matrix_dim + 4)

    # compute Lyapunov Exponential
    lyapex = nolds.lyap_e(data=v.astype(float), emb_dim=emb_dim, matrix_dim=matrix_dim, min_nb=min_nb, debug_data=False)
    largest_lyapex = lyapex[np.argmax(np.abs(lyapex))]
    return largest_lyapex


def spatialEntropy(xy, x_min, x_max, y_min, y_max):
    ''' proposed by Sergio A. Alvarez
    :param xy: xy position
    :param relative:
    :return: the entropy of heatmap
    '''
    est = kde(xy.transpose())

    xgrid, ygrid = np.mgrid[x_min:x_max:16j, y_min:y_max:16j]
    return entropy(np.array([est.pdf([x, y]) for (x, y) in zip(xgrid, ygrid)]).ravel()) \
           / np.log2(len(xgrid.ravel()))


def spectralEntropy(xy, fs=25):  # defaults to downsampled frequency
    ''' proposed by Sergio A. Alvarez
    :param xy: gaze - object series
    :param fs:
    :return:
    '''
    if len(xy) <= fs:
        return -1
    _, spx = welch(xy[:, 0], fs, nperseg=int(fs / 2))  # scipy.signal.welch
    _, spy = welch(xy[:, 1], fs, nperseg=int(fs / 2))  # equal spectrum discretization for x, y
    return entropy(spx + spy) / np.log2(len(_))  # scipy.stats.entropy


def computeLypanovMax(v, emb_dim=10):
    v[v == 0] = 1e-15
    if len(v) <= (emb_dim + 10):
        return np.nan
    v = (v - np.mean(v)) / np.std(v)
    return lyapunovExponent(v, emb_dim=emb_dim, matrix_dim=2)


def computeSampEn(v, emb_dim=10, r=0.4):
    v[v == 0] = 1e-15
    if len(v) <= (emb_dim + 5):
        return np.nan
    v = (v - np.mean(v)) / np.std(v)
    tolerance = r
    if nolds.sampen(v, emb_dim=emb_dim, tolerance=tolerance) == np.inf:
        print("error_se")
    return nolds.sampen(v, emb_dim=emb_dim, tolerance=tolerance)


def computeSkill(s, f):
    if len(f) > 0:
        s = s[~np.in1d(s[:, 1], f[:, 1])]

    n_s = len(s)
    n_f = len(f)

    skill = n_s / (n_s + n_f)

    return skill


def computeSequenceFeatures(s, f):
    if (len(f) <=1):
        return len(s), len(s)
    else:
        n_seq = []
        stop_seq = np.hstack([0, f[:, 1], s[-1, 1]])
        for i in range(len(stop_seq) - 1):
            start = stop_seq[i]
            stop = stop_seq[i + 1]
            n_seq.append(np.sum((s[:, 0] >= start) & (s[:, 1] <= stop)))

        return np.max(n_seq), np.average(n_seq)


def computePhaseSync(v1, v2, n=1, m=1):
    v1_analytic = hilbert(v1)
    v2_analytic = hilbert(v2)
    # v1_phase = np.arctan(v1_analytic.imag / v1_analytic.real)
    # v2_phase = np.arctan(v2_analytic.imag / v2_analytic.real)
    v1_phase = np.angle(v1_analytic)
    v2_phase = np.angle(v2_analytic)



    phase_diff = np.exp(1j * ( (n* v1_phase) -   (m * v2_phase)))
    # phase_diff = np.unwrap(v1_phase - v2_phase)

    avg_phase_diff = np.abs(np.average(phase_diff))

    print("phase_diff: "+ str(avg_phase_diff))

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)
    axs[0].plot(v1)
    axs[0].plot(v2)
    axs[1].plot(np.real(v1_analytic), np.imag(v1_analytic))
    axs[1].plot(np.real(v2_analytic), np.imag(v2_analytic))
    plt.show()

    return avg_phase_diff


def computeNormalizedED(v1, v2):
    dist = 0.5 * np.square(np.std(v1 - v2)) / (np.square(np.std(v1)) + np.square(np.std(v2)))
    return dist


def computeNormalizedCrossCorr(v1, v2, normalize=False, th=3):
    if normalize:
        v1 = (v1 - np.mean(v1)) / (np.std(v1))
        v2 = (v2 - np.mean(v2)) / (np.std(v2))


    cross_corr = signal.correlate(v1, v2, mode="full", method="direct") / len(v1)
    lags = signal.correlation_lags(len(v1), len(v2), mode="full")
    mid = np.argwhere(lags==0)[0, 0]
    # print(np.max(cross_corr))
    # # print(np.average(cross_corr[mid-th:mid+(th+1)]))
    # import matplotlib.pyplot as plt
    # plt.plot(v1)
    # plt.plot(v2)
    # plt.show()

    return np.max(cross_corr), lags[np.argmax(cross_corr)], np.average(cross_corr[mid-th:mid+(th+1)])

def computeMutualInf(v1, v2):
    try:
        return mutual_info(v1, v2)
    except:
        return np.nan

def computeTransferEntropy(v1, v2):
    try:
        return transfer_entropy(v1, v2, k=1)
    except:
        return np.nan


def computePhaseCrossCorr(v1, v2):
    v1_norm = (v1 - np.mean(v1)) / (np.std(v1))
    v2_norm = (v2 - np.mean(v2)) / (np.std(v2))
    # Compute the Fourier transforms of the signals
    x_fft = np.fft.fft(v1_norm)
    y_fft = np.fft.fft(v2_norm)

    # Compute the cross-correlation in the frequency domain
    cross_correlation_fft = x_fft * np.conj(y_fft)

    # Compute the inverse Fourier transform to get back to the time domain
    cross_correlation = np.fft.ifft(cross_correlation_fft)

    # Find the index of the maximum value in the cross-correlation
    max_index = np.argmax(np.abs(cross_correlation))

    # Calculate the phase shift
    num_points = len(v1)
    if max_index <= num_points // 2:
        phase_shift = 2 * np.pi * max_index / num_points
    else:
        phase_shift = 2 * np.pi * (max_index - num_points) / num_points

    return cross_correlation, phase_shift

def computeSpectralCoherence(v1, v2, fs=100, nperseg=16, bands=[0, 5, 20]):
    f, Cxy = signal.coherence(v1, v2, fs, nperseg=nperseg)
    # coherence = np.max(Cxy)
    # f_coherence = f[np.argmax(Cxy)]

    # low freq (0-15)
    # import matplotlib.pyplot as plt
    # plt.semilogy(f, Cxy)
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('Coherence')
    # plt.show()

    low_ch = np.average(Cxy[(f > bands[0]) & (f <= bands[1])])
    med_ch = np.average(Cxy[(f > bands[1]) & (f <= bands[2])])
    high_ch = np.average(Cxy[f > bands[2]])
    return low_ch, med_ch, high_ch, low_ch / high_ch


def computeScore(s, f, max_time=360., ball_trajetories=None, wall_trajectories=None):
    wall_x_min, wall_x_max = np.nanmin(wall_trajectories.filter(regex='_X').values), np.nanmax(
        wall_trajectories.filter(regex='_X').values)
    wall_z_min, wall_z_max = np.nanmin(wall_trajectories.filter(regex='_Z').values), np.nanmax(
        wall_trajectories.filter(regex='_Z').values)

    def computeVarMov(v):
        var_p1 = np.std(v[:, 2] - v[:, 0])
        var_p2 = np.std(v[:, 3] - v[:, 2])
        var_p3 = np.std(v[:, 1] - v[:, 3])

        avg_p1 = np.average(v[:, 2] - v[:, 0])
        avg_p2 = np.average(v[:, 3] - v[:, 2])
        avg_p3 = np.average(v[:, 1] - v[:, 3])

        return avg_p1, avg_p2, avg_p3, var_p1, var_p2, var_p3

    def computeRTSTD(v):
        signals = v[:, 1] - v[:, 0]
        if len(signals) <= 2:
            return 0
        return np.std(signals)

    def computeBallHullSTD(v, ball):

        # ball_x = (ball[v[:, 2], [0]] - wall_x_min) / (wall_x_max - wall_x_min)
        # ball_z = (ball[v[:, 2], [2]] - wall_z_min) / (wall_z_max - wall_z_min)
        ball_x = ball[v[:, 2], [0]]
        ball_z = ball[v[:, 2], [2]]
        ball_x_z = np.vstack([ball_x, ball_z]).transpose()

        if len(ball_x_z) <= 5:
            return 0, 0, -1, -1

        hull = ConvexHull(ball_x_z).volume
        std = np.average(np.std(ball_x_z, 0))
        spatial_entropy = spatialEntropy(ball_x_z, wall_x_min, wall_x_max, wall_z_min, wall_z_max)
        spectral_entropy = spectralEntropy(ball_x_z)

        return hull, std, spatial_entropy, spectral_entropy

    def computeSequenceFeatures(s, f):

        if (len(f) == 0) or (len(f) == 1):
            return len(s), len(s)
        else:
            n_seq = []
            stop_seq = np.hstack([0, f[:, 1], s[-1, 1]])
            for i in range(len(stop_seq) - 1):
                start = stop_seq[i]
                stop = stop_seq[i + 1]
                n_seq.append(np.sum((s[:, 0] >= start) & (s[:, 1] <= stop)))

            return np.max(n_seq), np.average(n_seq)

        # normalize

    avg_df = 0
    d_f = 0
    if len(f) > 0:
        s = s[~np.in1d(s[:, 1], f[:, 1])]
        avg_df = np.average(f[:, 1] - f[:, 0]) * 0.001
        d_f = np.sum(f[:, 1] - f[:, 0]) * 0.001

    avg_ds = np.average(s[:, 1] - s[:, 0]) * 0.001
    d_s = np.sum(s[:, 1] - s[:, 0]) * 0.001

    n_s = len(s)
    n_f = len(f)
    duration = d_s + d_f
    avg_episode = (avg_ds + avg_df) / 2

    # scores
    max_seq, avg_seq = computeSequenceFeatures(s, f)

    mix_score = (n_s - n_f) / (duration)
    # skill = n_s / avg_episode
    skill = n_s / (n_s + n_f)
    # task_score = duration / (n_f + 1)
    task_score = (n_s / (n_f + n_s)) * (d_s / duration)

    # ball
    bounce_hull, bounce_std, bounce_sp_entropy, bounce_sc_entropy = computeBallHullSTD(s, ball_trajetories)
    rt_lypanov = computeLypanovMax(s[:, 1] - s[:, 0], emb_dim=2)
    samp_en = computeSampEn(s[:, 1] - s[:, 0], emb_dim=2, r=0.55)
    std_rt = computeRTSTD(s)
    mov_avg1, mov_avg2, mov_avg3, mov_var1, mov_var2, mov_var3 = computeVarMov(s)

    return n_s, n_f, mix_score, skill, task_score, max_seq, avg_seq, bounce_hull, bounce_std, bounce_sp_entropy, bounce_sc_entropy, rt_lypanov, samp_en, std_rt, mov_avg1, mov_avg2, mov_avg3, mov_var1, mov_var2, mov_var3

def computeMultiDimSim(v1, v2):
    # v1 = stats.zscore(v1)
    # v2 = stats.zscore(v2)
   try:
        dtw_dist = metrics.dtw(v1, v2)
        if dtw_dist == np.inf:
            dtw_dist = np.nan
        lcss_dist = metrics.lcss(v1, v2, eps=0.5)

        return dtw_dist, lcss_dist
   except:
        return np.nan, np.nan

if __name__ == '__main__':
    trial1 = pd.read_pickle(
        "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-08_A\\2022-11-08_A_T02_complete.pkl")
    # trial2 = pd.read_pickle(
    #     "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-12-19_A\\2022-12-19_A_T06_complete.pkl")

    trial2 = pd.read_pickle(
        "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-12-08_A\\2022-12-08_A_T04A_complete.pkl")

    ball1 = trial1[2][0]

    ball2 = trial2[2][0]

    # print(computeScore(success_long, failure))
    computeScore(ball1["success_idx"], ball1["failures_idx"], max_time=len(ball1["trajectories"]) / 1000)
    computeScore(ball2["success_idx"], ball2["failures_idx"], max_time=len(ball2["trajectories"]) / 1000)
# if __name__ == '__main__':
#     v1 = np.random.random((30))
#     v2 = np.random.random((30)) + 10
#     v3 = np.random.random((30)) + 20
#
#     v = np.concatenate([v1, v2, v3])
#
#     vel, acc, saccade = computeVelAccV2(v)
#     import matplotlib.pyplot as plt
#
#     plt.plot(saccade)
#     plt.show()
#     print(vel.shape)
