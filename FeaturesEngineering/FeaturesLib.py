import nolds
import numpy as np
from scipy.signal import lfilter, convolve, savgol_filter
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde as kde, entropy
from scipy.signal import welch

def computeVelAccV2(v: np.array, normalize=True)-> np.array:
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
    kernel_vel = np.array([0, 1, 2, 3, 2, 1, 0]) # Duchowski's filter (https://doi.org/10.3758/BF03195486)
    velocity = computeSegmentAngles(v1, v2) * 100 # convert to deg / sec
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

    # import matplotlib.pyplot as plt
    # print(len(v))
    # print(len(velocity_norm))
    # print(len(acceler))
    # plt.plot(velocity)
    # plt.plot( lfilter(kernel_vel, 10, velocity))
    # plt.plot(np.convolve(velocity, kernel_vel/10, "same"))
    # plt.show()

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


def computeVelAcc(v):
    v1 = v[:-1]
    v2 = v[1:]
    speed = np.linalg.norm(v2 - v1, axis=-1)
    vel = np.sum(np.diff(v, n=1, axis=0), axis=-1)
    acc = np.diff(speed, n=1, axis=-1)

    return speed, vel, acc


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

            return lyapex[0]


def spatialEntropy(xy, x_min, x_max, y_min, y_max):
    ''' proposed by Sergio A. Alvarez
    :param xy: xy position
    :param relative:
    :return: the entropy of heatmap
    '''
    est = kde(xy.transpose())

    xgrid, ygrid = np.mgrid[x_min:x_max:16j, y_min:y_max:16j]
    return entropy(np.array([est.pdf([x,y]) for (x,y) in zip(xgrid, ygrid)]).ravel())\
           /np.log2(len(xgrid.ravel()))

def spectralEntropy(xy, fs=25):     # defaults to downsampled frequency
    ''' proposed by Sergio A. Alvarez
    :param xy: gaze - object series
    :param fs:
    :return:
    '''
    if len(xy) <= fs:
        return -1
    _, spx = welch(xy[:,0], fs, nperseg=int(fs/2))     # scipy.signal.welch
    _, spy = welch(xy[:,1], fs, nperseg=int(fs/2))     # equal spectrum discretization for x, y
    return entropy(spx + spy)/np.log2(len(_))  # scipy.stats.entropy


def computeScore(s, f, max_time = 360., ball_trajetories=None, wall_trajectories = None):

    wall_x_min, wall_x_max = np.nanmin(wall_trajectories.filter(regex='_X').values), np.nanmax(wall_trajectories.filter(regex='_X').values)
    wall_z_min, wall_z_max = np.nanmin(wall_trajectories.filter(regex='_Z').values), np.nanmax(wall_trajectories.filter(regex='_Z').values)
    def computeVarMov(v):
        var_p1 = np.std(v[:, 2] - v[:, 0])
        var_p2 = np.std(v[:, 3] - v[:, 2])
        var_p3 = np.std(v[:, 1] - v[:, 3])

        return var_p1, var_p2, var_p3

    def computeLypanovMax(v):
        signals =v[:, 1] - v[:, 0]
        if len(signals) <=20:
            return 0
        return lyapunovExponent(signals, emb_dim=10, matrix_dim=2)

    def computeSampEn(v):
        signals = v[:, 1] - v[:, 0]
        if len(signals) <= 20:
            return -1
        return nolds.sampen(s, emb_dim=10)

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

        if len(ball_x_z) <=5:
            return 0, 0, -1, -1

        hull = ConvexHull(ball_x_z).volume
        std = np.average(np.std(ball_x_z, 0))
        spatial_entropy = spatialEntropy(ball_x_z, wall_x_min, wall_x_max, wall_z_min, wall_z_max)
        spectral_entropy = spectralEntropy(ball_x_z)

        return hull, std, spatial_entropy, spectral_entropy



    def computeSequenceFeatures(s, f):

        if( len(f) == 0) or (len(f) == 1):
            return len(s), len(s)
        else:
            n_seq = []
            stop_seq = np.hstack([0, f[:, 1], s[-1, 1]])
            for i in range(len(stop_seq) - 1):
                start = stop_seq[i]
                stop =  stop_seq[i+1]
                n_seq.append(np.sum((s[:,0] >=start) & (s[:,1] <=stop)))

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

    mix_score = (n_s - n_f) / (duration)
    bounce_hull, bounce_std, bounce_sp_entropy, bounce_sc_entropy = computeBallHullSTD(s, ball_trajetories)
    rt_lypanov = computeLypanovMax(s)
    samp_en = computeSampEn(s)
    std_rt = computeRTSTD(s)
    mov_var1, mov_var2, mov_var3 = computeVarMov(s)
    skill = n_s / avg_episode
    task_score = duration / (n_f + 1)
    max_seq, avg_seq = computeSequenceFeatures(s, f)





    return n_s, n_f, mix_score, skill, task_score, max_seq, avg_seq, bounce_hull, bounce_std, bounce_sp_entropy, bounce_sc_entropy, rt_lypanov, samp_en, std_rt,  mov_var1, mov_var2, mov_var3



if __name__ == '__main__':

    trial1 = pd.read_pickle("F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-08_A\\2022-11-08_A_T02_complete.pkl")
    # trial2 = pd.read_pickle(
    #     "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-12-19_A\\2022-12-19_A_T06_complete.pkl")

    trial2 = pd.read_pickle(
        "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-12-08_A\\2022-12-08_A_T04A_complete.pkl")


    ball1 = trial1[2][0]

    ball2 = trial2[2][0]


    # print(computeScore(success_long, failure))
    computeScore(ball1["success_idx"], ball1["failures_idx"], max_time=len(ball1["trajectories"]) / 1000)
    computeScore(ball2["success_idx"], ball2["failures_idx"], max_time=len(ball2["trajectories"])/ 1000)
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