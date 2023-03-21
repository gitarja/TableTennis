import numpy as np
from scipy.interpolate import interp1d
from numpy import array
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
from Utils.Lib import movingAverage
from scipy.optimize import curve_fit

def interpolate(x: np.array, final: bool =False) -> np.array:
    '''
    interpolate x values and fill the extrapolation results with nan
    :param x: input
    :return: interpolated input

    baseInter1d will return nan value if exog inference is out of exog input (e.g exog input  2,3,4,5 and exog inference 1, 7, 8, 9)
    extrap1d will return value according to the formula
    '''
    def baseInter1d(interpolator, default_value=False):
        xs = interpolator.x
        ys = interpolator.y

        def pointwise(x):
            if x < xs[0] or x > xs[-1]:
                return np.nan
            else:
                return interpolator(x)

        def ufunclike(xs):
            return array(list(map(pointwise, array(xs))))

        return ufunclike

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

    # check the nan elements
    mask = np.isnan(x)
    if sum(mask) != 0:
        # if n input is less than 5 we cannot use cubic, not enough data for training. Just use linear and extrapolate the data with extrap1d
        # otherwise only interpolate data with qubic interpolation
        x_train = np.copy(x)
        if len(x_train) < 5:
            f = interp1d(np.flatnonzero(~mask), x_train[~mask], bounds_error=False, kind="linear", fill_value="extrapolate")
            f_x = extrap1d(f)
        else:
            if not final:
                x_train = movingAverage(x_train, 5)
            f = interp1d(np.flatnonzero(~mask), x_train[~mask], bounds_error=False, kind="linear", fill_value="extrapolate")
            f_x = baseInter1d(f, default_value=True)

        # fillin the nan values
        x[mask] = f_x(np.flatnonzero(mask))


    return x


def combineEpisode(ori: np.array, ep1: np.array, ep2: np.array, base:float, type=1) -> np.array:
    '''
    combine two episodes and crosschecked the combine version with the ori
    :param ori: the original episode
    :param ep1: the first episode that has been interpolated and extrapolated
    :param ep2: the second episode that has been interpolated and extrapolated
    :param base: the base to cut the intersection between the first and second episode (type 1= wall_y, type 2= ball_z)
    :param type: 1= combining first and second, 2 combining second and third
    :return: merged episodes
    '''
    if type==1:
        idx_ep1 = np.argwhere(ep1[:, 1] <= base)[-1][0] + 1
        idx_ep2 =  np.argwhere(ep2[:, 1] <= base)[0][0]
    else:
        idx_ep1 = np.argwhere(ep1[:, 2] >= base)[-1][0] + 1
        idx_ep2 = np.argwhere(ep2[:, 2] >= base)[0][0]

    merger_ele = np.empty((len(ori), 2, 3))
    merger_ele[:] = np.nan
    merger_ele[:idx_ep1, 0, :] = ep1[:idx_ep1]
    merger_ele[len(merger_ele) - len(ep2[idx_ep2:]):, 1, :] = ep2[idx_ep2:]

    conflict_area = np.argwhere(np.sum(np.isnan(np.sum(merger_ele, -1)), -1) == 0)
    merged_episodes = np.nanmean(merger_ele, 1)
    for c_idx in conflict_area:
        if type == 1:
            pol_idx = np.argmin(np.linalg.norm(merger_ele[c_idx, :, [0, 1]] - merged_episodes[c_idx - 1,  [0, 1]], axis=-1))
        else:
            pol_idx = np.argmin(np.linalg.norm(merger_ele[c_idx, :, [0, 2]] - merged_episodes[c_idx - 1,  [0, 2]], axis=-1))
        # pol_idx = np.argmin(np.linalg.norm(merger_ele[c_idx, :, 1:] - merged_episodes[c_idx-1, 1:], axis=-1))
        # print(pol_idx)
        merged_episodes[c_idx] = merger_ele[c_idx, pol_idx]




    return merged_episodes



def extrapolateAutoReg(data, idx_first_table=0, idx_ep=1):
    '''
    extrapolate data with AutoRegression.
    Some people serve by hitting to the table instead of the wall, in such case the first episode will be splitted into two.
    Extrapolation will be applied only on the second episode.
    :param data: data to be extrapolated
    :param idx_first_table:
    :return:
    '''
    x = np.copy(data)
    if idx_first_table != 0:
        x_prev = data[:idx_first_table]
        x = data[idx_first_table:]

    mask =  np.isnan(x)
    if sum(mask) != 0:
        # points_f = movingAverage(x[~mask], n=3)
        points_f = x[~mask]
        x[~mask] = points_f

        # second episode is going down, the others episode is going up
        if idx_ep == 2:
            model_f = AutoReg(endog=points_f, lags=1, trend="t") # going down
            model_f_fit = model_f.fit()

            points_b = np.flip(points_f)
            model_b = AutoReg(endog=points_b, lags=1, trend="ct") # going up
            model_b_fit = model_b.fit()
        else:
            model_f = AutoReg(endog=points_f, lags=1, trend="ct")  # going down
            model_f_fit = model_f.fit()

            points_b = np.flip(points_f)
            model_b = AutoReg(endog=points_b, lags=1, trend="t")  # going up
            model_b_fit = model_b.fit()

        # get the start and end of non-nan elements
        start, end = np.min(np.nonzero(~mask)), np.max(np.nonzero(~mask))

        idx_nan = np.nonzero(mask)[0]
        idx_inv = np.arange(len(x[:end+1]))[::-1]

        for idx in idx_nan:
            #forward extrapolation
            if idx > start:
                end_idx = len(points_f) + (idx - end)
                x[idx] = model_f_fit.predict(start=end_idx - len(idx_nan), end=end_idx, dynamic=True)[-1]
            #backward extrapolation
            else:
                end_idx = idx_inv[idx]
                x[idx] = model_b_fit.predict(start=end_idx - len(idx_nan), end=end_idx, dynamic=True)[-1]


    if idx_first_table != 0:
        # if first episode is splitted into two, combine them again
        x = np.concatenate([x_prev, x])

    return x



def extrapolateCurveFit(data, idx_first_table=0, idx_ep=1):
    '''
    extrapolate data with AutoRegression.
    Some people serve by hitting to the table instead of the wall, in such case the first episode will be splitted into two.
    Extrapolation will be applied only on the second episode.
    :param data: data to be extrapolated
    :param idx_first_table:
    :return:
    '''
    x = np.copy(data)
    if idx_first_table != 0:
        x_prev = data[:idx_first_table]
        x = data[idx_first_table:]

    mask =  np.isnan(x)
    if sum(mask) != 0:
        # points_f = movingAverage(x[~mask], n=3)
        points_f = x[~mask]
        x[~mask] = points_f

        # second episode is going down, the others episode is going up
        f = np.poly1d(np.polyfit(np.nonzero(~mask)[0], points_f, deg=1))

        idx_nan = np.nonzero(mask)[0]
        x[mask] = f(idx_nan)


    if idx_first_table != 0:
        # if first episode is splitted into two, combine them again
        x = np.concatenate([x_prev, x])

    return x






def cleanEpisodes(episode, ep1, ep2, ep3, end_ep2_idx, idx_first_table, wall_y, table_z):
    '''
    interpolate and extrapolate eac episode separately and combine them
    :param episode: a whole episode
    :param ep1: the first part of episode (racket -> wall)
    :param ep2: the second part of episode (wall -> table)
    :param ep3: the third part of episode (table -> racket)
    :param end_ep2_idx: end of the second episode
    :param idx_first_table:  if the participant server the ball by hitting it to the table, idx_first_table must be provided
    :return:
    '''

    ori_episode = np.copy(episode)

    x_inter_e1 = np.array([interpolate(ep1[:, i]) for i in range(3)]).transpose()
    x_inter_e2 = np.array([interpolate(ep2[:, i]) for i in range(3)]).transpose()
    x_inter_e3 = np.array([interpolate(ep3[:, i]) for i in range(3)]).transpose()



    x_inter_e1 = np.array([extrapolateAutoReg(x_inter_e1[:, i], idx_first_table) for i in range(3)]).transpose()
    x_inter_e2 = np.array([extrapolateAutoReg(x_inter_e2[:, i], idx_ep=2) for i in range(3)]).transpose()
    x_inter_e3 = np.array([extrapolateAutoReg(x_inter_e3[:, i]) for i in range(3)]).transpose()


    ep12 = combineEpisode(ori=episode[:end_ep2_idx], ep1=x_inter_e1, ep2=x_inter_e2, base=wall_y)



    episode[:end_ep2_idx] = ep12
    ep23 = combineEpisode(ori=episode, ep1=ep12, ep2=x_inter_e3, base=table_z, type=2)
    ep23 = np.array([interpolate(ep23[:, i], final=True) for i in range(3)]).transpose()
    ep23 = np.array([movingAverage(ep23[:, i], n=2) for i in range(3)]).transpose()

    # if np.std(ep23[:, 0]) < 20:
    #     ep23[:, 0] = movingAverage(ep23[:, 0], n=11)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(ori_episode[:, 0], ori_episode[:, 1], ori_episode[:, 2], c="black")
    #
    # ax.scatter(x_inter_e1[:, 0], x_inter_e1[:, 1], x_inter_e1[:, 2], c="blue")
    # ax.scatter(x_inter_e2[:, 0], x_inter_e2[:, 1], x_inter_e2[:, 2], c="orange")
    # ax.scatter(x_inter_e3[:, 0], x_inter_e3[:, 1], x_inter_e3[:, 2], c="red")

    ax.scatter(ep23[:, 0], ep23[:, 1], ep23[:, 2], c="green")
    plt.show()


    return ep23


if __name__ == '__main__':

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x = np.load("complete.npy")
    x_ori = np.copy(x)
    x[24:39] = np.nan
    x[:5] = np.nan
    x[66:72] = np.nan
    x_forward = np.copy(x[:39])
    x_backward = np.copy(x[24:72])
    x_backward2 = np.copy(x[66:])

    ax.scatter(x[:, 0], x[:, 1], x[:, 2])

    # perform interpolation
    x_inter_f = np.array([interpolate(x_forward[:, i]) for i in range(3)]).transpose()
    x_inter_b = np.array([interpolate(x_backward[:, i]) for i in range(3)]).transpose()
    x_inter_b2 = np.array([interpolate(x_backward2[:, i]) for i in range(3)]).transpose()

    x_inter_f = np.array([extrapolateAutoReg(x_inter_f[:, i]) for i in range(3)]).transpose()
    x_inter_b = np.array([extrapolateAutoReg(x_inter_b[:, i]) for i in range(3)]).transpose()
    x_inter_b2 = np.array([extrapolateAutoReg(x_inter_b2[:, i]) for i in range(3)]).transpose()

    ep12 = combineEpisode(ori=x[:72], ep1=x_inter_f, ep2=x_inter_b, base=1900.93)
    x[:72] = ep12
    ep23 = combineEpisode(ori=x, ep1=ep12, ep2=x_inter_b2, base=751.247, type=2)
    ep23 = np.array([interpolate(ep23[:, i]) for i in range(3)]).transpose()
    # ep23 = np.array([movingAverage(ep23[:, i], n=2) for i in range(3)]).transpose()

    # c = np.diff(x_interextra, axis=0)
    # d = np.diff(x_inter_f, axis=0)

    # ax.scatter(ep23[:, 0], ep23[:, 1], ep23[:, 2], c="orange")


    # ax.scatter(x_inter_f[:, 0], x_inter_f[:, 1], x_inter_f[:, 2], c="red")
    # ax.scatter(x_inter_b[:, 0], x_inter_b[:, 1], x_inter_b[:, 2], c="green")
    # ax.scatter(x_inter_b2[:, 0], x_inter_b2[:, 1], x_inter_b2[:, 2], c="purple")



    plt.show()
