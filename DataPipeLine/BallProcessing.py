import numpy as np
import ezc3d
import pandas as pd
from scipy.spatial import Delaunay
from Utils.Valleys import findValleys, groupValleys, checkValleysSanity, removeSpecialValleyTable
from Utils.DataReader import ViconReader, SubjectObjectReader

from Utils.Interpolate import cleanEpisodes
from scipy.spatial import distance_matrix
from Utils.Lib import movingAverage, interPolateDistance, savgolFilter
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from Config.Episodes import EpisodesParamsSingle, EpisodesParamsDouble
import pickle


class BallProcessing:
    '''
    A class that contains functions to find and clean (interpolate and extrapolate) ball trajectories.
    Since the unlabelled trajectories can be not only ball, but also Tobii reflection of other things.
    We need to filter out unlabelled data oustide the region of interest (wall and table) and data near Tobii glasses
    '''

    # # relocated table
    # ball_area = np.array([
    #     [-749.966797, -1017.712341, 726.281189],  # table pt1_x - 60, table pt1_y - 1500, table pt1_z
    #     [817.196533, -1004.012634, 726.281189],  # table pt4_x  - 60, table pt4_y - 1500, table pt4_z
    #     [-800.386292, 2000.592773, 726.281189],  # table pt3_x, table pt3_y + 600, table pt3_z
    #     [927.946838, 2000.623779, 726.281189],  # table pt2_x, table pt2_y + 600, table pt2_z
    #
    #     [-749.966797, 217.712341, 2036.201416],  # table pt1_x  - 60, table pt1_y, table pt1_z * 2
    #     [817.196533, 204.012634, 2036.201416],  # table pt4_x  + 60, table pt4_y, table pt4_z * 2
    #     [-800.061218, 2000.592773, 2036.201416],  # wall pt4_x - 50, wall pt4_y, wall pt4_z + 400
    #     [927.275452, 2000.623779, 2036.201416],  # wall pt1_x + 50, wall pt1_y, wall pt1_z + 400
    #
    # ])

    def __init__(self, obj: list, sub: list, session_name: str, double=False):
        '''
        :param obj: list of objects
        :param sub: list of subjects
        '''

        self.racket_1 = None
        self.racket_2 = None
        self.wall_mean = None
        self.table_mean = None
        self.session_name = session_name
        self.rackets = []
        for o in obj:
            if o["name"] == 'Racket1' or o["name"] == 'Racket1a':
                self.racket_1 = o
                self.rackets.append(o)
            elif o["name"] == 'Racket2' or o["name"] == 'Racket2a':
                self.racket_2 = o
                self.rackets.append(o)
            elif o["name"] == 'Wall':
                self.wall_mean = np.nanmean(o["trajectories"], 0)
                self.wall_centro = np.nanmean(self.wall_mean.reshape(4, 3), 0)
            elif o["name"] == 'Table':
                self.table_mean = np.nanmean(o["trajectories"], 0)
                self.table_centro = np.nanmean(self.table_mean.reshape(4, 3), 0)

        if double:
            self.racket_1 = self.rackets[0]
            self.racket_2 = self.rackets[1]
        else:
            if (self.racket_1 is None) & (self.racket_2 is not None):
                self.racket_1 = self.racket_2
        tr = self.normalizeTable(self.table_mean.reshape(4, 3))

        wr = self.wall_mean.reshape((4, 3))
        tp = tr[[0, -1]]
        wp = wr[[0, -1]]
        self.table_area = np.array([
            [tp[0, 0], tp[0, 1], tp[0, 2]],  # pt1
            [tp[1, 0], tp[1, 1], tp[1, 2]],  # pt4
            [tp[0, 0], wp[0, 1], tp[0, 2]],  # pt2'
            [tp[1, 0], wp[1, 1], tp[1, 2]],  # pt3'

            [tp[0, 0], tp[0, 1], tp[0, 2] + 1000],  # pt1'
            [tp[1, 0], tp[1, 1], tp[1, 2] + 1000],  # pt4'
            [tp[0, 0], wp[0, 1], tp[0, 2] + 1000],  # pt2''
            [tp[1, 0], wp[1, 1], tp[1, 2] + 1000],  # pt3''

        ])  # get yposition of pt1 and p4

        self.ball_area = np.array([
            [tr[0, 0] - 400, tr[0, 1] - 1600, tr[0, 2] - 200],  # table pt1_x - 60, table pt1_y - 1500, table pt1_z
            [tr[3, 0] + 400, tr[0, 1] - 1600, tr[0, 2] - 200],  # table pt4_x  - 60, table pt1_y - 1500, table pt1_z
            [tr[0, 0] - 200, wr[0, 1] + 10, tr[0, 2] - 30],  # table pt1_x, table pt3_y + 800, table pt3_z
            [tr[3, 0] + 200, wr[0, 1] + 10, tr[0, 2] - 30],  # table pt4_x, table pt2_y + 800, table pt2_z

            [tr[0, 0] - 400, tr[0, 1] - 1600, wr[0, 2] * 2.7],  # table pt1_x  - 60, table pt1_y, table pt1_z * 2
            [tr[3, 0] + 400, tr[0, 1] - 1600, wr[0, 2] * 2.7],  # table pt4_x  + 60, table pt4_y, table pt4_z * 2
            [tr[0, 0] - 200, wr[0, 1] + 10, wr[0, 2] * 2.7],  # wall pt4_x - 50, wall pt4_y, wall pt4_z + 400
            [tr[3, 0] + 200, wr[0, 1] + 10, wr[0, 2] * 2.7],  # wall pt1_x + 50, wall pt1_y, wall pt1_z + 400

        ])

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1, projection='3d')
        #
        # ax.scatter(self.table_area[:, 0], self.table_area[:, 1], self.table_area[:, 2])
        # ax.scatter(self.ball_area[:, 0], self.ball_area[:, 1], self.ball_area[:, 2])
        # plt.show()

        self.subjects = []
        for s in sub:
            self.subjects.append(s)

    def normalizeTable(self, table):
        pt1 = table[0]
        pt2 = table[1]
        pt3 = table[2]
        pt4 = table[3]

        def swap(a, b):
            return b, a

        if (pt1[0] > 0):
            pt1, pt4 = swap(pt1, pt4)
            pt2, pt3 = swap(pt2, pt3)
        if (pt1[1] > pt3[1]):
            pt1, pt3 = swap(pt1, pt3)
            pt2, pt4 = swap(pt2, pt4)

        return np.vstack([pt1, pt2, pt3, pt4])




    def extrapolateInterpolateBall(self, ball, success_episodes, failure_ep, valleys_w, valleys_t, table, wall,
                                   th_t=150, th_failure_extrapolate=400):
        '''
        Exxtrapolate and interpolate the success episodes
        :param ball: ball trajectories
        :param success_episodes: sucess episodes
        :param valleys_w: valleys wall index
        :param valleys_t: valleys table index
        :param table: table centroid
        :param table: wall centroid
        :return: interpolated and extrapolated ball trajectory
        '''
        table_z = table[2] - 15
        wall_y = wall[1] + 15
        i = 0

        for i in range(len(failure_ep) - 1):
            f_now = failure_ep[i]
            f_next = failure_ep[i + 1]
            if (f_next - f_now) <= th_failure_extrapolate:
                # print(str(f_now) + " " + str(f_next))
                success_episodes = np.append(success_episodes, np.array([[f_now, f_next]]), axis=0)

        success_episodes = success_episodes[success_episodes[:, 1].argsort()]
        # for 2022-12-20_A_T04
        # success_episodes = np.delete(success_episodes, np.argwhere(success_episodes[:, 0] == 16406), 0)
        # success_episodes = np.delete(success_episodes, np.argwhere(success_episodes[:, 0] == 28786), 0)

        # for 2023-02-15_M_T03
        # success_episodes = np.delete(success_episodes, np.argwhere(success_episodes[:, 0] == 11673), 0)

        # for 2023-02-15_M_T06
        # success_episodes = np.delete(success_episodes, np.argwhere(success_episodes[:, 0] == 19887), 0)
        # success_episodes = np.delete(success_episodes, np.argwhere(success_episodes[:, 0] == 28053), 0)

        # for 2022-11-15_A_T01
        # success_episodes = np.delete(success_episodes, np.argwhere(success_episodes[:, 0] == 23610), 0)

        for s in success_episodes:
            print(s)

            i += 1
            episode = ball[s[0]:s[1]]
            mask = np.isnan(episode[:, 0])

            if np.sum(mask) > 0:
                # filter out false point
                # if one point is not close to any other points, exclude it from the trajectory

                distance = distance_matrix(episode, episode, p=2)
                distance[np.diag_indices_from(distance)] = 999999
                min_dist = np.nanmin(distance, 0)
                min_dist[min_dist == 999999] = np.nan
                episode[min_dist >= th_t] = np.nan


                if (len(np.where((valleys_w >= s[0]) & (valleys_w <= s[1]))[0]) == 0) or (
                        len(np.where((valleys_t >= s[0]) & (valleys_t <= s[1]))[0]) == 0):
                    continue
                # index of wall valley in one episode
                idx_valley_w = int(valleys_w[np.where((valleys_w >= s[0]) & (valleys_w <= s[1]))[0][0]])
                # index of table valley in one episode
                ib_valleys_t = valleys_t[np.where((valleys_t >= s[0]) & (valleys_t <= s[1]))[0]]
                idx_valley_t = int(ib_valleys_t[0])

                idx_first_table = 0
                # if there are two table valleys in one episode
                if (len(ib_valleys_t) > 1):
                    if (idx_valley_w > ib_valleys_t[0]):
                        idx_first_table = int(ib_valleys_t[0]) - s[0]
                        idx_valley_t = int(ib_valleys_t[1])

                # normalize the position of wall valley
                idx_valley_w = idx_valley_w - s[0]

                # reset valley wall
                if np.isnan(np.sum(episode[idx_valley_w - 1])) | np.isnan(np.sum(episode[idx_valley_w + 1])):
                    episode[idx_valley_w] = np.nan

                # normalize the position of table valley


                idx_valley_t = idx_valley_t - s[0]

                # reset valley table
                # if np.isnan(np.sum(episode[idx_valley_t - 1])) | np.isnan(np.sum(episode[idx_valley_t + 1])):
                #     episode[idx_valley_t] = np.nan

                # mask of episode
                mask = np.isnan(episode[:, 1])

                # get the value of the valley
                valley_w = episode[idx_valley_w - 1:idx_valley_w + 1]
                valley_t = episode[idx_valley_t - 1:idx_valley_t + 1]

                e_ep1 = idx_valley_w + 1
                if np.isnan(np.sum(episode[idx_valley_w + 1])):
                    e_ep1 = e_ep1 + 1
                s_ep2 = idx_valley_w
                e_ep2 = idx_valley_t + 1
                if np.isnan(np.sum(episode[idx_valley_t + 1])):
                    e_ep2 = e_ep2 + 1
                s_ep3 = idx_valley_t

                # first valley wall
                # decide the end of the ep1 and the start of ep 2
                if (np.sum(np.isnan(valley_w)) != 0) | (np.abs(valley_w[0, 1] - wall_y) > 250):
                    bf_valley = np.nonzero(~mask[:idx_valley_w])[0]
                    bf_valley = bf_valley[bf_valley > 0]
                    af_valley = np.nonzero(~mask[idx_valley_w:idx_valley_t])[0]
                    af_valley = af_valley[af_valley > 0]
                    s_ep2 = bf_valley[np.argmax(bf_valley - idx_valley_w)] + 1
                    if (len(af_valley)) != 0:
                        e_ep1 = af_valley[np.argmin(af_valley - idx_valley_w)] + idx_valley_w

                # first valley table
                # decide the end of the ep2 and the start of ep3
                if (np.sum(np.isnan(valley_t)) != 0) | (np.abs(valley_t[0, 2] - table_z) > 250):
                    bf_valley = np.nonzero(~mask[s_ep2:idx_valley_t])[0]
                    af_valley = np.nonzero(~mask[idx_valley_t:])[0]
                    if (len(bf_valley)) != 0:
                        s_ep3 = bf_valley[np.argmax(bf_valley - idx_valley_t)] + s_ep2 + 1
                    e_ep2 = af_valley[np.argmin(af_valley - idx_valley_t)] + idx_valley_t

                # split the episodes
                # for 2022-11-14_M_T03
                # if s[0] == 18533:
                #     e_ep2 = e_ep2 - 6

                # for 2022-12-21_M_T05
                # if s[0] == 33372:
                #     e_ep2 = e_ep2 - 9


                # for 2022-11-15_M_T06
                # if s[0] == 26553:
                #     e_ep2 = e_ep2 - 6
                #     s_ep3 = s_ep3 - 6

                # for 2022-11-15_A_T02
                # if s[0] == 10607:
                #     episode[56] = (episode[55] + episode[57]) / 2
                #     episode[58] = (episode[57] + episode[59]) / 2
                #     episode[59] = (episode[58] + episode[59]) / 2
                #     episode[59, 1] = episode[59, 1] - 10
                #     episode[59, 2] = episode[59, 2] - 10
                #
                #     episode[69] = (episode[70] + episode[71]) / 2
                #     episode[69, 1] = episode[69, 1] + 10
                #     episode[69, 2] =  episode[69, 2] -40
                #     e_ep2 = e_ep2 - 1
                #     s_ep3 = s_ep3 - 1

                ep1 = episode[:e_ep1]
                ep2 = episode[s_ep2:e_ep2]
                ep3 = episode[s_ep3:]

                # clean episodes

                if (len(ep1) != 0) & (len(ep2) != 0) & (len(ep3) != 0):
                    # print("cleaned")
                    clean_episode = cleanEpisodes(episode, ep1, ep2, ep3, e_ep2, idx_first_table, wall_y, table_z)
                else:


                    clean_episode = episode

                    # fig = plt.figure()
                    # ax = fig.add_subplot(projection='3d')
                    #
                    # ax.scatter(episode[:, 0], episode[:, 1], episode[:, 2], c="green")
                    #
                    # plt.show()
            else:
                # if nothing to interpolate or extrapolate, just do moving average to smooth the trajectory
                # print(np.sum(np.isnan(episode[:, 0])))
                clean_episode = np.array([movingAverage(episode[:, i], n=2) for i in range(3)]).transpose()

                # fig = plt.figure()
                # ax = fig.add_subplot(projection='3d')
                #
                # ax.scatter(clean_episode[:, 0], clean_episode[:, 1], clean_episode[:, 2], c="green")
                #
                # plt.show()


            ball[s[0]:s[1]] = clean_episode
            # for 2023-01-26_M_T03
            # ball[10127] = (ball[10126] + ball[10128]) / 2

            # for 2022-11-15_M_T06
            # ball[23005] = (ball[23004] + ball[23006]) / 2




        return ball



    def maskInterpolateSanity(self, ball_tj, n, th):
        area_mask = Delaunay(self.ball_area).find_simplex(ball_tj) >= 0
        nan_mask = np.isnan(ball_tj)[:, 0]
        mask = np.copy(area_mask)
        th = int(th * n)
        for i in range(0, len(mask), 1):
            if (np.sum(area_mask[i:i + n]) >= th) & (area_mask[i] == False):
                mask[i] = True
            else:
                mask[i] = False

        return mask



    def filteringUnLabData(self, data, tobii_data, th_tracking=100):
        '''
        filter out unlabelled data that are not ball trajectories e.g. noise from Tobii
        :param data:
        :param tobii_data:
        :return:
        '''
        # transform matrix
        data = data.transpose((-1, 1, 0))
        tobii_data = tobii_data.transpose((1, 0, -1))

        data = data[:len(tobii_data), :, :]

        tobii_dist_1 = np.linalg.norm(data - np.expand_dims(tobii_data[:, 0, :], 1), axis=-1)
        tobii_m = (tobii_dist_1 < 200)
        if tobii_data.shape[1] >= 2:
            tobii_dist_2 = np.linalg.norm(data - np.expand_dims(tobii_data[:, 1, :], 1), axis=-1)
            tobii_m = (tobii_dist_1 < 200) | (tobii_dist_2 < 200)

        data[tobii_m] = np.nan
        # remove all Tobii ghost data
        # check whether points inside the polygon
        inside_outside = Delaunay(self.ball_area).find_simplex(data) >= 0
        has_ball_bool = inside_outside[:, np.argwhere(np.sum(inside_outside, 0) > 0)]
        has_ball = np.squeeze(data[:, np.argwhere(np.sum(inside_outside, 0) > 0), :])
        positive_count = np.sum(has_ball_bool, 1)
        only_one_cand = np.nonzero(positive_count == 1)[0]  # only one candidate ball
        more_one_cand = np.nonzero(positive_count > 1)[0]  # more than one candidate ball

        # normalize ball traj
        ball_tj = np.empty((len(data), 3))
        ball_tj[:] = np.nan

        # fill in the ball trajectories
        ball_tj[only_one_cand] = has_ball[only_one_cand, np.nonzero(has_ball_bool[only_one_cand])[1]]

        std_ball_cand = np.average(np.nanstd(has_ball[more_one_cand], 1), 1)
        std_ball_idx = np.nonzero(std_ball_cand < 30)[0]  # ghost ball
        ball_tj[more_one_cand[std_ball_idx]] = np.nanmean(has_ball[more_one_cand[std_ball_idx]], 1)

        # if there are more than one ball, track it
        std_ball_multiple_idx = np.nonzero(std_ball_cand > 30)[0]  # happens when there are multiple unlabel dataset

        ball_tj_mask = np.nonzero(~np.isnan(np.sum(ball_tj, 1)))[0]
        for idx in more_one_cand[std_ball_multiple_idx]:
            # find the nearet not a nan
            nearest_idx = ball_tj_mask[np.argmin(np.abs(ball_tj_mask - idx))]
            ball_cand_dist = np.linalg.norm(has_ball[idx] - np.expand_dims(ball_tj[nearest_idx], 0), axis=-1)
            cond = ball_cand_dist[~np.isnan(ball_cand_dist)] < th_tracking
            if np.sum(cond) != 0:
                ball_tj[idx] = has_ball[idx][np.nanargmin(ball_cand_dist)]
                ball_tj_mask = np.nonzero(~np.isnan(np.sum(ball_tj, 1)))[0]

        return ball_tj

