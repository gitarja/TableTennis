import numpy as np
import ezc3d
from scipy.spatial import Delaunay
import transforms3d
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from Utils.Interpolate import extrap1d

from numpy.lib import stride_tricks


class BallPreProcessing:


    def __init__(self, obj, sub):

        for o in obj:
            if o["name"] == 'Racket1':
                self.racket_1 = o
            elif o["name"] == 'Racket2':
                self.racket_2 = o
            elif o["name"] == 'Wall':
                self.wall_mean = np.nanmean(o["trajectories"], 0)
            elif o["name"] == 'Table':
                self.table_mean = np.nanmean(o["trajectories"], 0)


        self.subjects = []
        for s in sub:
            self.subjects.append(s)

    def plotting3D(self, data):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data[17500:17700, 0], data[17500:17700, 1], data[17500:17700, 2])
        plt.show()

    def plotPerAxis(self, before, after):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(6)

        before = before[15000:15900]
        after = after[15000:15900]
        # x-axis
        axs[0].plot(np.arange(len(before)), before[:, 0])
        axs[1].plot(np.arange(len(before)), after[:, 0])
        # y-axis
        axs[2].plot(np.arange(len(before)), before[:, 1])
        axs[3].plot(np.arange(len(before)), after[:, 1])

        # z-axis
        axs[4].plot(np.arange(len(before)), before[:, 2])
        axs[5].plot(np.arange(len(before)), after[:, 2])
        plt.show()

    def plotNearTable(self, before, after):

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2)

        y_after = after[:, 1]

        axs[0].plot(np.arange(len(before)), before[:, 1])
        axs[1].plot(np.arange(len(before)), after[:, 1])


    def normalizeData(self, data, tobii_data):
        # ball_area = np.array([
        #     [-793.04703234, 500.82939009, 732.0650296], # table pt1_x, table pt1_y, table pt1_z
        #     [774.58230762, 500.82939009, 749.83444349], # table pt4_x, table pt4_y, table pt4_z
        #     [-766.44288645, 2407.78545434, 742.19295307], # table pt3_x, table pt3_y + 600, table pt3_z
        #     [797.48482387, 2327.98461769, 761.78510793], # table pt2_x, table pt2_y + 600, table pt2_z
        #
        #     [-793.04703234, 1007.82939009, 1666.323361],# table pt1_x, table pt1_y, table pt1_z * 2
        #     [774.58230762, 955.18120783, 1642.58802859],# table pt4_x, table pt4_y, table pt4_z * 2
        #     [-688.01776107, 2407.78545434, 2000.58802859],# wall pt4_x, wall pt4_y, wall pt4_z + 400
        #     [876.92080693, 2348.65824315, 2000.58802859],# wall pt1_x, wall pt1_y, wall pt1_z + 400
        #
        # ])



        # transform matrix
        data = data.transpose((-1, 1, 0))
        tobii_data =  tobii_data.transpose((-1, 1, 0))
        tobii_dist_1 = np.linalg.norm(data - np.expand_dims(tobii_data[:, 0, :], 1), axis=-1)
        tobii_m = (tobii_dist_1 < 300)
        if tobii_data.shape[1] > 2:
            tobii_dist_2 = np.linalg.norm(data - np.expand_dims(tobii_data[:, 1, :], 1), axis=-1)
            tobii_m = (tobii_dist_1 < 300) | (tobii_dist_2 < 300)

        data[tobii_m] = np.nan
        # remove all Tobii ghost data



        # check whether points inside the polygon
        inside_outside = Delaunay(self.ball_area).find_simplex(data) >= 0
        has_ball_bool = inside_outside[:, np.argwhere(np.sum(inside_outside, 0) > 0)]
        has_ball = np.squeeze(data[:, np.argwhere(np.sum(inside_outside, 0) > 0), :])
        positive_count = np.sum(has_ball_bool, 1)
        only_one_cand = np.nonzero(positive_count == 1)[0] # only one candidate ball
        more_one_cand = np.nonzero(positive_count > 1)[0] # more than one candidate ball
        # normalize ball traj
        ball_tj = np.empty((len(data), 3))
        ball_tj[:] = np.nan

        # fill in the ball trajectories
        ball_tj[only_one_cand] = has_ball[only_one_cand, np.nonzero(has_ball_bool[only_one_cand])[1]]


        std_ball_cand = np.average(np.nanstd(has_ball[more_one_cand], 1), 1)
        std_ball_idx = np.nonzero(std_ball_cand < 100)[0]

        ball_tj[more_one_cand[std_ball_idx]] = np.nanmean(has_ball[more_one_cand[std_ball_idx]], 1)
        # ball_tj_diff = np.sqrt(np.sum(np.power(np.diff(ball_tj, axis=0), 2), -1))
        # m = np.argwhere(ball_tj_diff > 50)
        # ball_tj[m+1, :] = np.nan

        # perform sliding avg

        # ball_tj = np.array([self.avgMoving(ball_tj[:, i], n=10) for i in range(3)]).transpose()


        return ball_tj





    def findTheBall(self, x):
        if np.average(np.std(x)) <0.05:
            return np.average(x, 1)
        else:
            a = np.empty((1, 3))
            a[:] = np.nan
            return a

    def maskInterpolateSanity(self, ball_tj, n, th):
        area_mask = Delaunay(self.ball_area).find_simplex(ball_tj) >= 0
        nan_mask = np.isnan(ball_tj)[:, 0]
        mask = np.copy(area_mask)
        th = int(th * n)
        for i in range(0, len(mask), 1):
            if (np.sum(area_mask[i:i+n]) >= th) & (area_mask[i] == False):
                mask[i] = True
            else:
                mask[i] = False

        return mask

    def interPolate(self, data, n=300, th=0.8):
        # copy data
        dup_data = np.copy(data)
        # convert threshold to the number of frame
        th = int(th * n)
        # nan masking
        bool_mask = np.isnan(data)

        for i in range(0, len(data), n):
            mask = bool_mask[i:i + n]
            # print(np.sum(mask))
            if (np.sum(mask) <= th) & (np.sum(mask) > 0) & (len(mask) > th):
                snipset = dup_data[i:i + n]

                f = interp1d(np.flatnonzero(~mask), snipset[~mask], bounds_error=False, kind="quadratic")
                f_x = extrap1d(f, default_value=True)
                snipset[mask] = f_x(np.flatnonzero(mask))

                # f = InterpolatedUnivariateSpline(np.flatnonzero(~mask), snipset[~mask],k=2)
                # snipset[mask] = f(np.flatnonzero(mask))

                dup_data[i:i + n] = snipset

        return dup_data

    def avgMoving(self, x, n=3):

        cumsum = np.nancumsum(np.insert(x, 0, 0))
        return (cumsum[n:] - cumsum[:-n]) / float(n)

    def avgSmoothing(self, x, n=3):
        if n % 2 != 1:
            raise Exception("n must be an odd number")
        mid = int((n / 2) + 1)
        a = stride_tricks.sliding_window_view(x, n)
        mean_a = np.average(a, -1)
        mean_a = np.where(a[:, mid] == np.NaN, mean_a, a[:, mid])

        return mean_a


    def readData(self, file_path: str = None):
        data = ezc3d.c3d(file_path)
        labels = data['parameters']['POINT']['LABELS']['value']
        unlabeled_idx = [i for i in range(len(labels)) if
                         "*" in labels[i]]  # the column label of the unlabelled marker starts with *
        data_points = np.array(data['data']['points'])

        unlabelled_data = data_points[0:3, unlabeled_idx, :]

        normalized_data = self.normalizeData(unlabelled_data, tobii_data)
        normalized_data = np.array([self.avgSmoothing(normalized_data[:, i], n=11) for i in range(3)]).transpose()