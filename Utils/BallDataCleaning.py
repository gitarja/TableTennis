import numpy as np
import ezc3d
from scipy.spatial import Delaunay
import transforms3d
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from Utils.Interpolate import extrap1d
from scipy.ndimage import label
from numpy.lib import stride_tricks
from Utils.DataReader import ViconReader
from scipy.signal import butter, lfilter, freqz

class BallFinding:
    # relocated table
    ball_area = np.array([
        [-749.966797, -1017.712341, 726.281189],  # table pt1_x - 60, table pt1_y - 1500, table pt1_z
        [817.196533, -1004.012634, 746.193665],  # table pt4_x  - 60, table pt4_y - 1500, table pt4_z
        [-749.386292, 1860.348145, 739.174377],  # table pt3_x, table pt3_y + 600, table pt3_z
        [814.946838, 1860.348145, 739.174377],  # table pt2_x, table pt2_y + 600, table pt2_z

        [-749.966797, 217.712341, 2036.201416],  # table pt1_x  - 60, table pt1_y, table pt1_z * 2
        [817.196533, 204.012634, 2036.201416],  # table pt4_x  - 60, table pt4_y, table pt4_z * 2
        [-690.061218, 1947.592773, 2036.201416],  # wall pt4_x, wall pt4_y, wall pt4_z + 400
        [877.275452, 1930.623779, 2036.201416],  # wall pt1_x, wall pt1_y, wall pt1_z + 400

    ])

    def __init__(self, obj, sub):

        self.racket_1 = None
        self.racket_2 = None
        self.wall_mean = None
        self.table_mean = None

        for o in obj:
            if o["name"] == 'Racket1' or o["name"] == 'Racket1a':
                self.racket_1 = o
            elif o["name"] == 'Racket2':
                self.racket_2 = o
            elif o["name"] == 'Wall':
                self.wall_mean = np.nanmean(o["trajectories"], 0)
                self.wall_centro = np.nanmean(self.wall_mean.reshape(4, 3), 0)
            elif o["name"] == 'Table':
                self.table_mean = np.nanmean(o["trajectories"], 0)
                self.table_centro = np.nanmean(self.table_mean.reshape(4, 3), 0)


        if (self.racket_1 is None ) & (self.racket_2 is not None ):
            self.racket_1 = self.racket_2

        self.subjects = []
        for s in sub:
            self.subjects.append(s)

    def plotting3D(self, data):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data[17500:17700, 0], data[17500:17700, 1], data[17500:17700, 2])
        plt.show()

    def plotRacket(self, ball, racket1, racket2=None):
        import matplotlib.pyplot as plt
        ball = ball[0:5000]
        racket1 = racket1[0:5000]

        dist =  np.linalg.norm(ball-racket1, axis=-1)

        fig, axs = plt.subplots(1)
        axs.plot(np.arange(len(dist)), dist)
        if racket2 is not None:
            racket2 = racket2[0:5000]
            dist2 = np.linalg.norm(ball - racket2, axis=-1)
            axs.plot(np.arange(len(dist2)), dist2)

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
        tobii_data =  tobii_data.transpose((1, 0, -1))

        data = data[:len(tobii_data),:, :]

        tobii_dist_1 = np.linalg.norm(data - np.expand_dims(tobii_data[:, 0, :], 1), axis=-1)
        tobii_m = (tobii_dist_1 < 300)
        if tobii_data.shape[1] >= 2:
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



    def findEpisodes(self, ball, r1, r2=None, wall=None, table=None, th_s=0.3, th_d=200):


        '''
        :param ball: ball trajectory
        :param r1: racket 1 trajectory
        :param r2: racket 2 trajectory
        :return:
        '''

        # ball = ball[0:6300]
        # r1 = r1[0:6300]

        def interPolate(x):

            mask = np.isnan(x)
            if np.sum(mask) > 0:
                x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])
            return x

        def findValleys(dist, th_s, th_d):

            # find the peak with template matching, create the cos template
            t = np.linspace(-1.5 * np.pi, -0.5 * np.pi, 3)
            qrs_filter = np.cos(t)

            # normalize data
            dist_norm = (dist - np.nanmean(dist)) / np.nanstd(dist)

            # calculate cross correlation
            similarity = np.correlate(dist_norm, qrs_filter, mode="same")
            similarity = similarity / np.nanmax(similarity)

            valleys = np.nonzero((similarity > th_s) & (dist <= th_d))[0]

            return valleys

        def groupValleys(p, dist, width_th=10, n_group=(1, 4)):
            '''
            The peak detection algorithm finds multiple peaks for each QRS complex.
            Here we group collections of peaks that are very near (within threshold) and we take the median index
            '''
            # initialize output
            output = np.empty(0)

            # label groups of sample that belong to the same peak
            valley_groups, num_groups = label(np.diff(p) < width_th)

            # iterate through groups and take the mean as peak index
            for i in np.unique(valley_groups)[1:]:
                valley_group = p[np.where(valley_groups == i)]
                # output = np.append(output, valley_group[np.argmin(dist[valley_group])])
                output = np.append(output, valley_group[np.argmin(dist[valley_group])])
                # if (len(valley_group) >= n_group[0]) & (len(valley_group) <= n_group[1]):
                #     output = np.append(output, valley_group[np.argmin(dist[valley_group])])
            return output

        def groupEpisodes(idx, wall_vy=None, table_vy=None, th=150):

            sucess_idx = []
            failure_idx = []
            i = 0
            while i < (len(idx) - 1):
                check_wall_valley = (wall_vy > idx[i]) & (wall_vy < idx[i + 1])
                if (idx[i + 1] - idx[i] < th) & np.sum(check_wall_valley) > 0:
                    curr_wall = wall_vy[((wall_vy > idx[i]) & (wall_vy < idx[i + 1]))][-1]
                    check_table_valley = np.sum((table_vy > curr_wall) & (table_vy < idx[i + 1])) == 1
                    if check_table_valley:
                        sucess_idx.append([idx[i], idx[i + 1]])
                    else:
                        failure_idx.append(idx[i])
                    # i+=1
                else:
                    failure_idx.append(idx[i])
                i += 1
            success = np.vstack(sucess_idx).astype(int)
            failures = np.array(failure_idx).astype(int)
            # check failure sanity
            mask = np.nonzero(np.diff(failures) < 100)[0] + 1
            failures = np.delete(failures, mask)
            return success, failures

        def checkValleysSanity(racket_vy, wall_vy, dis_th=100):
            sane_valleys = []
            prev = 0
            for i in range(len(racket_vy) - 1):
                dist_r_r = np.abs(prev - racket_vy[i])
                prev = racket_vy[i]
                if dist_r_r > dis_th:
                    if np.sum((wall_vy > racket_vy[i]) & (wall_vy < racket_vy[i + 1])) >= 1:
                        sane_valleys.append(racket_vy[i])
                else:
                    sane_valleys.append(racket_vy[i])

            sane_valleys.append(racket_vy[len(racket_vy) - 1])

            return np.array(sane_valleys)

        if r2 is not None:
            r2 = r2[10000:29000]
            r = np.concatenate([np.expand_dims(r1, 1), np.expand_dims(r2, 1)], 1)
            ball = np.expand_dims(ball, 1)
            dist_rackets = np.linalg.norm(ball - r, axis=-1)
            dist_rackets[:, 0] = interPolate(dist_rackets[:, 0])
            dist_rackets[:, 1] = interPolate(dist_rackets[:, 1])

            # get peaks racket 1
            peaks1 = findValleys(dist_rackets[:, 0], th_s, th_d)
            peaks1 = groupValleys(peaks1, dist_rackets[:, 0], 11)

            # get peaks racket 2

            peaks2 = findValleys(dist_rackets[:, 1], th_s, th_d)

            peaks2 = groupValleys(peaks2, dist_rackets[:, 1], 11)

            import matplotlib.pyplot as plt
            plt.plot(np.arange(len(dist_rackets)), dist_rackets[:, 0], label="dist", color="#66c2a5", linewidth=1)
            plt.plot(np.arange(len(dist_rackets)), dist_rackets[:, 1], label="dist", color="#fc8d62", linewidth=1)
            plt.plot(peaks1, np.repeat(70, peaks1.shape[0]), label="peaks", color="orange", marker="o",
                     linestyle="None")
            plt.plot(peaks2, np.repeat(70, peaks2.shape[0]), label="peaks", color="green", marker="o",
                     linestyle="None")
            # plt.ylim([0, 200])
            plt.show()

            # success_ep, failure_ep = groupDoubleEpisodes(peaks1, peaks2)
        else:
            dist_rackets = interPolate(np.linalg.norm(ball - r1, axis=-1))

            dist_walll = interPolate(np.abs(ball[:, 1] - wall[1]))
            dist_table = interPolate(np.abs(ball[:, 2] - table[2]))
            # get peaks racket 1
            valleys_wall = findValleys(dist_walll, th_s, th_d=200)
            valleys_wall = groupValleys(valleys_wall, dist_walll, width_th= 11, n_group=(1, 50))

            valleys_table = findValleys(dist_table, th_s, th_d=100)
            valleys_table = groupValleys(valleys_table, dist_table, width_th=11, n_group=(1, 50))

            valleys1 = findValleys(dist_rackets, th_s=0.5, th_d=150)
            valleys1 = groupValleys(valleys1, dist_rackets, width_th =50, n_group=(1, 150))

            # check valley sanity
            valleys1 = checkValleysSanity(valleys1, valleys_wall)
            success_ep, failure_ep = groupEpisodes(valleys1, valleys_wall, valleys_table, th=250)
            import matplotlib.pyplot as plt
            plt.plot(np.arange(len(dist_rackets)), dist_rackets, label="dist", color="#51A6D8", linewidth=1)
            plt.plot(np.arange(len(dist_walll)), dist_walll, label="dist wall", color="#fc8d62", linewidth=1)
            plt.plot(np.arange(len(dist_table)), dist_table, label="dist wall", color="#8da0cb", linewidth=1)
            plt.plot(valleys_table, np.repeat(70, valleys_table.shape[0]), label="peaks", color="blue", marker="o",
                     linestyle="None", alpha=0.5)
            plt.plot(valleys_wall, np.repeat(70, valleys_wall.shape[0]), label="peaks", color="black", marker="o",
                     linestyle="None", alpha=0.5)
            plt.plot(success_ep[:, 0], np.repeat(70, success_ep.shape[0]), label="peaks", color="orange", marker="o", linestyle="None", alpha=0.5)
            plt.plot(success_ep[:, 1], np.repeat(70, success_ep.shape[0]), label="peaks", color="green", marker="o", linestyle="None", alpha=0.5)
            plt.plot(failure_ep, np.repeat(70, failure_ep.shape[0]), label="peaks", color="red", marker="o",
                     linestyle="None", alpha=0.5)
            # plt.plot(valleys_wall, np.repeat(70, valleys_wall.shape[0]), label="peaks", color="blue", marker="o",
            #          linestyle="None", alpha=0.5)
            plt.show()



    def butter_lowpass(self, cutoff, fs, order=5):
        return butter(order, cutoff, fs=fs, btype='low', analog=False, output="ba")

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

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
        mask = np.isnan(x)
        smooth_x =  np.convolve(x, np.ones(n), 'same') / n
        smooth_x[mask] = np.nan
        return smooth_x

    def avgSmoothing(self, x, n=3):
        if n == 0:
            return x
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
        tobii_data = []
        for s in self.subjects:
            tobii_data.append(s["segments"][:, -3:]) # tobii data are the 3 last columns
        tobii_data = np.array(tobii_data)
        normalized_data = self.normalizeData(unlabelled_data, tobii_data)
        smooth_ball_data = np.array([self.avgMoving(normalized_data[:, i], n=1) for i in range(3)]).transpose()
        smooth_r1_data = np.array([self.avgMoving(self.racket_1["segments"][:, 3+i], n=1) for i in range(3)]).transpose()



        if self.racket_2 != None:
            smooth_r2_data = np.array(
                [self.avgMoving(self.racket_2["segments"][:, 3 + i], n=1) for i in range(3)]).transpose()
            # self.plotRacket(smooth_ball_data, smooth_r1_data, smooth_r2_data)
            # self.findEpisodes(smooth_ball_data, smooth_r1_data, smooth_r2_data)
        # self.plotRacket(smooth_ball_data, smooth_r1_data)
        self.findEpisodes(smooth_ball_data, smooth_r1_data, wall=self.wall_centro, table=self.table_mean, th_s=0.3)



if __name__ == '__main__':



    # reader = ECGReader()
    # ecg_files = ["F:\\users\\prasetia\\data\\TableTennis\\Pilot\\27-07-2022\\ECG\\rp-0\\20220727_095702516448.csv",
    #              "F:\\users\\prasetia\\data\\TableTennis\\Pilot\\27-07-2022\\ECG\\rp-1\\20220727_095702413999.csv"]
    # data = reader.extractData(ecg_files)
    # print(data)

    # reader = TobiiReader()
    # ecg_files = ["F:\\users\\prasetia\\data\\TableTennis\\Experiments\\08.11.2022-afternoon\\Tobii\\Data Export - 08-11-2022-afternoo\\08-11-2022-afternoo Recording 1 (2).tsv",
    #              "F:\\users\\prasetia\\data\\TableTennis\\Experiments\\08.11.2022-afternoon\\Tobii\\Data Export - 08-11-2022-afternoo\\08-11-2022-afternoo Recording 1 (3).tsv"]
    # data = reader.extractData(ecg_files)
    # print(data)


    reader = ViconReader()
    obj, sub = reader.extractData("F:\\users\\prasetia\\data\\TableTennis\\Test\\T04.csv")


    reader = BallFinding(obj, sub)
    data = reader.readData("F:\\users\\prasetia\\data\\TableTennis\\Test\\T04.c3d")
    # data = reader.extractData("F:\\users\\prasetia\\data\\TableTennis\\Pilot\\Ball\\BallTest08.c3d")