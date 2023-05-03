import numpy as np
import ezc3d
import pandas as pd
from scipy.spatial import Delaunay
from Utils.Valleys import findValleys, groupValleys, checkValleysSanity
from Utils.DataReader import ViconReader

from Utils.Interpolate import cleanEpisodes
from scipy.spatial import distance_matrix
from Utils.Lib import movingAverage, interPolateDistance
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from Config.Episodes import EpisodesParamsSingle, EpisodesParamsDouble

class BallFinding:
    '''
    A class that contains functions to find and clean (interpolate and extrapolate) ball trajectories.
    Since the unlabelled trajectories can be not only ball, but also Tobii reflection of other things.
    We need to filter out unlabelled data oustide the region of interest (wall and table) and data near Tobii glasses
    '''
    # relocated table
    ball_area = np.array([
        [-749.966797, -1017.712341, 726.281189],  # table pt1_x - 60, table pt1_y - 1500, table pt1_z
        [817.196533, -1004.012634, 726.281189],  # table pt4_x  - 60, table pt4_y - 1500, table pt4_z
        [-800.386292, 2000.592773, 726.281189],  # table pt3_x, table pt3_y + 600, table pt3_z
        [927.946838, 2000.623779, 726.281189],  # table pt2_x, table pt2_y + 600, table pt2_z

        [-749.966797, 217.712341, 2036.201416],  # table pt1_x  - 60, table pt1_y, table pt1_z * 2
        [817.196533, 204.012634, 2036.201416],  # table pt4_x  + 60, table pt4_y, table pt4_z * 2
        [-800.061218, 2000.592773, 2036.201416],  # wall pt4_x - 50, wall pt4_y, wall pt4_z + 400
        [927.275452, 2000.623779, 2036.201416],  # wall pt1_x + 50, wall pt1_y, wall pt1_z + 400

    ])

    def __init__(self, obj: list, sub: list):
        '''
        :param obj: list of objects
        :param sub: list of subjects
        '''

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

        if (self.racket_1 is None) & (self.racket_2 is not None):
            self.racket_1 = self.racket_2

        tp = self.table_mean.reshape((4, 3))[[0, -1]]
        wp = self.wall_mean.reshape((4, 3))[[0, -1]]
        self.table_area = np.array([
        [tp[0, 0], tp[0, 1], tp[0, 2]],  # pt1
        [tp[1, 0], tp[1, 1], tp[1, 2]],  # pt4
        [wp[0, 0], wp[0, 1], wp[0, 2]],  # pt2'
        [wp[1, 0], wp[1, 1], wp[1, 2]],  # pt3'

        [tp[0, 0], tp[0, 1], tp[0, 2] + 1000],  # pt1'
        [tp[1, 0], tp[1, 1], tp[1, 2] + 1000],  # pt4'
        [wp[0, 0], wp[0, 1], wp[0, 2] + 1000],  # pt2''
        [wp[1, 0], wp[1, 1], wp[1, 2] + 1000],  # pt3''

    ])  # get yposition of pt1 and p4
        self.subjects = []
        for s in sub:
            self.subjects.append(s)

    def extrapolateInterpolateBall(self, ball, success_episodes, failure_ep, valleys_w, valleys_t, table, wall, th_t=150, th_failure_extrapolate=400):
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
            f_next = failure_ep[i+1]
            if (f_next - f_now) <= th_failure_extrapolate:
                # print(str(f_now) + " " + str(f_next))
                success_episodes = np.append(success_episodes, np.array([[f_now, f_next]]), axis=0)

        success_episodes = success_episodes[success_episodes[:, 1].argsort()]
        for s in success_episodes:
            print(s)
            i+=1
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

                # index of wall valley in one episode
                if len(np.where((valleys_w >= s[0]) & (valleys_w <= s[1]))[0]) == 0:
                    continue
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
                if np.isnan(np.sum(episode[idx_valley_w - 1])) & np.isnan(np.sum(episode[idx_valley_w + 1])):
                    episode[idx_valley_w] = np.nan






                # normalize the position of table valley
                idx_valley_t = idx_valley_t - s[0]

                # reset valley table
                # if np.isnan(np.sum(episode[idx_valley_t - 1])) & np.isnan(np.sum(episode[idx_valley_t + 1])):
                #     episode[idx_valley_t] = np.nan


                # mask of episode
                mask = np.isnan(episode[:, 1])

                # get the value of the valley
                valley_w = episode[idx_valley_w-1:idx_valley_w+1]
                valley_t = episode[idx_valley_t-1:idx_valley_t+1]

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
                if np.sum(np.isnan(valley_w)) != 0:
                    bf_valley = np.nonzero(~mask[:idx_valley_w])[0]
                    af_valley = np.nonzero(~mask[idx_valley_w:idx_valley_t])[0]
                    s_ep2 = bf_valley[np.argmax(bf_valley - idx_valley_w)]  + 1
                    e_ep1 = af_valley[np.argmin(af_valley - idx_valley_w)] + idx_valley_w

                # first valley table
                # decide the end of the ep2 and the start of ep3
                if np.sum(np.isnan(valley_t)) != 0:
                    bf_valley = np.nonzero(~mask[s_ep2:idx_valley_t])[0]
                    af_valley = np.nonzero(~mask[idx_valley_t:])[0]
                    s_ep3 = bf_valley[np.argmax(bf_valley - idx_valley_t)] + s_ep2 + 1
                    e_ep2 = af_valley[np.argmin(af_valley - idx_valley_t)] + idx_valley_t

                # split the episodes
                ep1 = episode[:e_ep1]
                ep2 = episode[s_ep2:e_ep2]
                ep3 = episode[s_ep3:]

                # clean episodes
                clean_episode = cleanEpisodes(episode, ep1, ep2, ep3, e_ep2, idx_first_table, wall_y, table_z)
            else:
                # if nothing to interpolate or extrapolate, just do moving average to smooth the trajectory
                # print(np.sum(np.isnan(episode[:, 0])))
                clean_episode = np.array([movingAverage(episode[:, i], n=2) for i in range(3)]).transpose()

            ball[s[0]:s[1]] = clean_episode

        return ball

    def filteringUnLabData(self, data, tobii_data, th_tracking = 100):
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
        only_one_cand = np.nonzero(positive_count == 1)[0]  # only one candidate ball
        more_one_cand = np.nonzero(positive_count > 1)[0]  # more than one candidate ball

        # normalize ball traj
        ball_tj = np.empty((len(data), 3))
        ball_tj[:] = np.nan

        # fill in the ball trajectories
        ball_tj[only_one_cand] = has_ball[only_one_cand, np.nonzero(has_ball_bool[only_one_cand])[1]]

        std_ball_cand = np.average(np.nanstd(has_ball[more_one_cand], 1), 1)
        std_ball_idx = np.nonzero(std_ball_cand < 30)[0] # ghost ball
        ball_tj[more_one_cand[std_ball_idx]] = np.nanmean(has_ball[more_one_cand[std_ball_idx]], 1)

        # if there are more than one ball, track it
        std_ball_multiple_idx = np.nonzero(std_ball_cand > 30)[0] # happens when there are multiple unlabel dataset

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

    def findEpisodesSingle(self, ball, r1, wall=None, table=None, params:EpisodesParamsSingle=None):

        '''
        :param ball: ball trajectory
        :param r1: racket 1 trajectory
        :param r2: racket 2 trajectory
        :return:
        '''

        def groupEpisodes(idx, wall_vy=None, table_vy=None, th=150, th_failure=400, th_failure_sanity=100):
            # check whether the ball inside the table or not
            inside_outside_table = Delaunay(self.table_area).find_simplex(ball) >= 0
            valleys_table_outside = valleys_table[(inside_outside_table[valleys_table.astype(int)] == False)]
            sucess_idx = []
            failure_idx = []
            i = 0
            while i < (len(idx) - 1):
                check_wall_valley = (wall_vy > idx[i]) & (wall_vy < idx[i + 1])
                if (idx[i + 1] - idx[i] < th) & np.sum(check_wall_valley) > 0:
                    curr_wall = wall_vy[((wall_vy > idx[i]) & (wall_vy < idx[i + 1]))][-1]
                    table_in_episode = (table_vy > curr_wall) & (table_vy < idx[i + 1])
                    check_table_valley = np.sum(table_in_episode) == 1
                    if check_table_valley:
                        sucess_idx.append([idx[i], idx[i + 1]])
                    else:
                        table_last = table_vy[table_in_episode == True][-1]
                        # double table valley (one inside table and the other is outside)
                        if np.isin(table_last, valleys_table_outside):
                            sucess_idx.append([idx[i], idx[i + 1]])
                        else:
                            print("failure")
                            failure_idx.append(idx[i])
                    # i+=1
                else:
                    failure_idx.append(idx[i])
                i += 1
            success = np.vstack(sucess_idx).astype(int)
            failures = np.array(failure_idx).astype(int)
            # check failure sanity
            mask = np.nonzero(np.diff(failures) < th_failure_sanity)[0] + 1
            failures = np.delete(failures, mask)

            # if ball bounce twice but the player still continue
            success_start = success[:, 0]
            delete_idx = []
            for i in range(len(failures)):
                f_start = failures[i]
                f_stop = failures[i + 1] if i + 1 < len(failures) else success[-1][1]
                s_b_f = success_start[(success_start > f_start) & (success_start < f_stop)]
                if len(s_b_f) > 0:
                    if (s_b_f[0] - f_start) < th_failure:
                        sbf_idx = np.nonzero((success_start > f_start) & (success_start < f_stop))[0]
                        delete_idx.append(sbf_idx)
            if len(delete_idx) != 0:
                delete_idx = np.concatenate(delete_idx)
                failures = np.append(failures, success_start[delete_idx])
                success = np.delete(success, delete_idx, axis=0)

            return success, failures

        dist_rackets = interPolateDistance(np.linalg.norm(ball - r1, axis=-1))
        dist_walll = interPolateDistance(np.abs(ball[:, 1] - wall[1]))
        dist_table = interPolateDistance(np.abs(ball[:, 2] - table[2]))
        # get valleys wall
        valleys_wall = findValleys(dist_walll, th_c=params.TH_CONFIDENCE, th_d=params.TH_D_WALL)
        valleys_wall = groupValleys(valleys_wall, dist_walll, within_th=params.TH_WITHIN, n_group=(1, 50))
        # get valleys table
        valleys_table = findValleys(dist_table, th_c=params.TH_CONFIDENCE, th_d=params.TH_D_TABLE)
        valleys_table = groupValleys(valleys_table, dist_table, within_th=params.TH_WITHIN, n_group=(1, 50))
        # get valleys racket 1
        valleys_rackets = findValleys(dist_rackets, th_c=params.TH_CONFIDENCE, th_d=params.TH_D_RACKET)
        valleys_rackets = groupValleys(valleys_rackets, dist_rackets, within_th=params.TH_WITHIN_RACKET, n_group=(1, 150))

        # check valley sanity
        valleys_rackets = checkValleysSanity(valleys_rackets, valleys_wall)
        success_ep, failure_ep = groupEpisodes(valleys_rackets, valleys_wall, valleys_table, th=params.TH_SUCCESS_EPISODES, th_failure_sanity=params.TH_FAILURE_SANITY)
        failure_ep = np.sort(failure_ep)
        # import matplotlib.pyplot as plt
        # plt.plot(np.arange(len(dist_rackets)), dist_rackets, label="dist", color="#66c2a5", linewidth=1)
        # plt.plot(np.arange(len(dist_walll)), dist_walll, label="dist wall", color="#8da0cb", linewidth=1)
        # plt.plot(np.arange(len(dist_table)), dist_table, label="dist wall", color="#e78ac3", linewidth=1)
        #
        # plt.plot(valleys_table, np.repeat(70, valleys_table.shape[0]), label="peaks", color="blue", marker="o",
        #          linestyle="None", alpha=0.5)
        # plt.plot(valleys_wall, np.repeat(70, valleys_wall.shape[0]), label="peaks", color="black", marker="o",
        #          linestyle="None", alpha=0.5)
        # plt.plot(success_ep[:, 0], np.repeat(70, success_ep.shape[0]), label="peaks", color="green", marker="o",
        #          linestyle="None", alpha=0.5)
        # plt.plot(success_ep[:, 1], np.repeat(70, success_ep.shape[0]), label="peaks", color="green", marker="o",
        #          linestyle="None", alpha=0.5)
        # plt.plot(failure_ep, np.repeat(70, failure_ep.shape[0]), label="peaks", color="red", marker="o",
        #          linestyle="None", alpha=0.5)
        # plt.plot(valleys_wall, np.repeat(70, valleys_wall.shape[0]), label="peaks", color="blue", marker="o",
        #          linestyle="None", alpha=0.5)
        # plt.show()

        return success_ep, failure_ep, valleys_rackets, valleys_wall, valleys_table

    def findEpisodesDouble(self, ball, r1, r2, wall=None, table=None, params:EpisodesParamsDouble=None):

        def groupEpisodes(idx1, idx2, wall_vy=None, table_vy=None, th=150, th_failure=400, th_failure_sanity=100):
            # check whether the ball inside the table or not
            inside_outside_table = Delaunay(self.table_area).find_simplex(ball) >= 0
            valleys_table_outside = valleys_table[(inside_outside_table[valleys_table.astype(int)] == False)]
            sucess_idx = []
            failure_idx = []
            i = 0
            idx = np.sort(np.concatenate([idx1, idx2]))
            while i < len(idx) - 1:
                # print(str(idx[i]) + " " + str(idx[i + 1]))
                check_wall_valley = (wall_vy > idx[i]) & (wall_vy < idx[i + 1])
                # check whether two valleys belong to the same person or not
                check_diff_sub = not (np.isin(idx[i], idx1) & np.isin(idx[i + 1], idx1)) | (
                            np.isin(idx[i], idx2) & np.isin(idx[i + 1],
                                                            idx2))  # check whether the valley come from different subjects
                if (idx[i + 1] - idx[i] < th) & (np.sum(check_wall_valley) > 0) & check_diff_sub:
                    curr_wall = wall_vy[((wall_vy > idx[i]) & (wall_vy < idx[i + 1]))][-1]
                    table_in_episode = (table_vy > curr_wall) & (table_vy < idx[i + 1])
                    check_table_valley = np.sum(table_in_episode) == 1
                    # there must be one valley table between valley wall and the next valley racket
                    if check_table_valley:
                        sucess_idx.append([idx[i], idx[i + 1]])
                    else:
                        if np.sum(table_in_episode) > 1:
                            table_last = table_vy[table_in_episode == True][-1]
                            if np.isin(table_last, valleys_table_outside):
                                sucess_idx.append([idx[i], idx[i + 1]])
                            else:
                                # print("failure")
                                failure_idx.append(idx[i])
                        else:
                            # print("failure")
                            failure_idx.append(idx[i])
                    # i+=1
                else:
                    # print("failure")
                    failure_idx.append(idx[i])
                i += 1

            success = np.vstack(sucess_idx).astype(int)
            failures = np.array(failure_idx).astype(int)

            # check failure sanity, two consecutive failures within less than one second (100 frames) is not possible
            mask = np.nonzero(np.diff(failures) < th_failure_sanity)[0] + 1
            failures = np.delete(failures, mask)

            # if fails but the player still continue
            success_start = success[:, 0]
            delete_idx = []
            for i in range(len(failures)):
                f_start = failures[i]
                f_stop = failures[i + 1] if i + 1 < len(failures) else success[-1][1]
                s_b_f = success_start[(success_start > f_start) & (success_start < f_stop)]
                if len(s_b_f) > 0:
                    if (s_b_f[0] - f_start) < th_failure:
                        curr_wall = wall_vy[((wall_vy > s_b_f[0]) & (wall_vy < f_start))]
                        table_in_episode = table_vy[(table_vy > s_b_f[0]) & (table_vy < f_start)]
                        if len(curr_wall) + len(table_in_episode) != 2:
                            sbf_idx = np.nonzero((success_start > f_start) & (success_start < f_stop))[0]
                            delete_idx.append(sbf_idx)
            if len(delete_idx) != 0:
                delete_idx = np.concatenate(delete_idx)
                failures = np.append(failures, success_start[delete_idx])
                success = np.delete(success, delete_idx, axis=0)

            return success, failures



        dist_rackets1 = interPolateDistance(np.linalg.norm(ball - r1, axis=-1))
        dist_rackets1[np.isnan(np.sum(r1, 1))] = 10000
        dist_rackets2 = interPolateDistance(np.linalg.norm(ball - r2, axis=-1))
        dist_rackets2[np.isnan(np.sum(r2, 1))] = 10000
        dist_walll = interPolateDistance(np.abs(ball[:, 1] - wall[1]))
        dist_table = interPolateDistance(np.abs(ball[:, 2] - table[2]))


        # save all distances
        # add_text = ""
        # np.savetxt("dist_racket1_"+add_text+".csv", dist_rackets1, delimiter=",")
        # np.savetxt("dist_racket2_" + add_text + ".csv", dist_rackets2, delimiter=",")
        # np.savetxt("dist_walll_" + add_text + ".csv", dist_walll, delimiter=",")
        # np.savetxt("dist_table_" + add_text + ".csv", dist_table, delimiter=",")

        # print("Min Dist Racket 1:" + str(np.min(dist_rackets1)))
        # print("Min Dist Racket 2:" + str(np.min(dist_rackets2)))

        # get valleys racket 1
        valleys_rackets1 = findValleys(dist_rackets1, th_c=params.TH_CONFIDENCE, th_d=params.TH_D_RACKET)
        valleys_rackets1 = groupValleys(valleys_rackets1, dist_rackets1, within_th=params.TH_WITHIN_RACKET)

        # get valleys racket 2
        valleys_rackets2 = findValleys(dist_rackets2, th_c=params.TH_CONFIDENCE, th_d=params.TH_D_RACKET)
        valleys_rackets2 = groupValleys(valleys_rackets2, dist_rackets2, within_th=params.TH_WITHIN_RACKET)

        # get valleys wall
        valleys_wall = findValleys(dist_walll, th_c=params.TH_CONFIDENCE, th_d=params.TH_D_WALL)
        valleys_wall = groupValleys(valleys_wall, dist_walll, within_th=params.TH_WITHIN)
        # get valleys table
        valleys_table = findValleys(dist_table, th_c=params.TH_CONFIDENCE, th_d=params.TH_D_TABLE)
        valleys_table = groupValleys(valleys_table, dist_table, within_th=params.TH_WITHIN)


        # check valley sanity
        valleys_rackets1 = checkValleysSanity(valleys_rackets1, valleys_wall, dis_th=params.TH_RACKET_SANITY)
        valleys_rackets2 = checkValleysSanity(valleys_rackets2, valleys_wall, dis_th=params.TH_RACKET_SANITY)

        success_ep, failure_ep = groupEpisodes(valleys_rackets1, valleys_rackets2, valleys_wall, valleys_table, th=params.TH_SUCCESS_EPISODES,
                                               th_failure=params.TH_FAILURE_MID_EPISODES, th_failure_sanity=params.TH_FAILURE_SANITY)

        failure_ep = np.sort(failure_ep)
        plt.plot(np.arange(len(dist_rackets1)), dist_rackets1, label="dist", color="#238b45", linewidth=1)
        plt.plot(np.arange(len(dist_walll)), dist_walll, label="dist wall", color="#8da0cb", linewidth=1)
        plt.plot(np.arange(len(dist_table)), dist_table, label="dist table", color="#e78ac3", linewidth=1)
        plt.plot(np.arange(len(dist_rackets2)), dist_rackets2, label="dist", color="#66c2a4", linewidth=1)

        plt.plot(valleys_wall, np.repeat(20, valleys_wall.shape[0]), label="peaks", color="blue", marker="o",
                 linestyle="None", alpha=0.5)
        plt.plot(valleys_table, np.repeat(20, valleys_table.shape[0]), label="peaks", color="orange", marker="o",
                 linestyle="None", alpha=0.5)


        plt.plot(valleys_rackets1, np.repeat(20, valleys_rackets1.shape[0]), label="peaks", color="black", marker="o",
                 linestyle="None", alpha=0.5)
        plt.plot(valleys_rackets2, np.repeat(20, valleys_rackets2.shape[0]), label="peaks", color="black", marker="o",
                 linestyle="None", alpha=0.5)

        plt.plot(success_ep[:, 0], np.repeat(20, success_ep.shape[0]), label="peaks", color="green", marker="o",
                 linestyle="None", alpha=0.5)
        plt.plot(success_ep[:, 1], np.repeat(20, success_ep.shape[0]), label="peaks", color="green", marker="o",
                 linestyle="None", alpha=0.5)
        plt.plot(failure_ep, np.repeat(70, failure_ep.shape[0]), label="peaks", color="red", marker="o",
                 linestyle="None", alpha=0.5)

        plt.show()

        return success_ep, failure_ep, valleys_rackets1, valleys_rackets2, valleys_wall, valleys_table

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

    def cleanSingleData(self, file_path: str = None):
        data = ezc3d.c3d(file_path)
        labels = data['parameters']['POINT']['LABELS']['value']
        unlabeled_idx = [i for i in range(len(labels)) if
                         "*" in labels[i]]  # the column label of the unlabelled marker starts with *
        data_points = np.array(data['data']['points'])

        unlabelled_data = data_points[0:3, unlabeled_idx, :]
        tobii_data = []
        for s in self.subjects:
            tobii_data.append(s["segments"][:, -3:])  # tobii data are the 3 last columns
        tobii_data = np.array(tobii_data)
        normalized_data = self.filteringUnLabData(unlabelled_data, tobii_data)
        smooth_ball_data = np.array([movingAverage(normalized_data[:, i], n=1) for i in range(3)]).transpose()
        smooth_r1_data = np.array(
            [movingAverage(self.racket_1["segments"][:, 3 + i], n=1) for i in range(3)]).transpose()

        success_ep, failure_ep, valleys_rackets, valleys_wall, valleys_table = self.findEpisodesSingle(smooth_ball_data,
                                                                                                       smooth_r1_data,
                                                                                                       wall=self.wall_centro,
                                                                                                       table=self.table_mean
                                                                                                       ,params=EpisodesParamsSingle("not_clean"))
        clean_ball = self.extrapolateInterpolateBall(smooth_ball_data, success_ep, failure_ep, valleys_wall, valleys_table,
                                                     wall=self.wall_centro,
                                                     table=self.table_mean, th_failure_extrapolate=EpisodesParamsSingle.TH_FAILURE_EXTRAPOLATE)



        success_ep2, failure_ep2, valleys_rackets2, valleys_wall2, valleys_table2 = self.findEpisodesSingle(clean_ball,
                                                                                                            smooth_r1_data,
                                                                                                            wall=self.wall_centro,
                                                                                                            table=self.table_mean,
                                                                                                            params=EpisodesParamsSingle("clean_ball"))
        print("Before cleaning")
        print("Success: " + str(len(success_ep)))
        print("Failure: " + str(len(failure_ep)))

        print("After cleaning")
        print("Success: " + str(len(success_ep2)))
        print("Failure: " + str(len(failure_ep2)))




        return clean_ball

    def cleanDoubleData(self, file_path: str = None):
        data = ezc3d.c3d(file_path)
        labels = data['parameters']['POINT']['LABELS']['value']
        unlabeled_idx = [i for i in range(len(labels)) if
                         "*" in labels[i]]  # the column label of the unlabelled marker starts with *
        data_points = np.array(data['data']['points'])

        unlabelled_data = data_points[0:3, unlabeled_idx, :]
        tobii_data = []
        for s in self.subjects:
            tobii_data.append(s["segments"][:, -3:])  # tobii data are the 3 last columns

        tobii_data = np.array(tobii_data)
        normalized_data = self.filteringUnLabData(unlabelled_data, tobii_data)

        smooth_ball_data = np.array([movingAverage(normalized_data[:, i], n=1) for i in range(3)]).transpose()
        smooth_r1_data = np.array(
            [movingAverage(self.racket_1["segments"][:, 3 + i], n=1) for i in range(3)]).transpose()
        smooth_r2_data = np.array(
            [movingAverage(self.racket_2["segments"][:, 3 + i], n=1) for i in range(3)]).transpose()

        success_ep, failure_ep, valleys_rackets1, valleys_rackets2, valleys_wall, valleys_table = self.findEpisodesDouble(smooth_ball_data,
                                                                                                       smooth_r1_data,
                                                                                                       smooth_r2_data,
                                                                                                       wall=self.wall_centro,
                                                                                                       table=self.table_mean
                                                                                                       ,params=EpisodesParamsDouble("not_clean"))

        clean_ball = self.extrapolateInterpolateBall(smooth_ball_data, success_ep, failure_ep, valleys_wall, valleys_table,
                                                     wall=self.wall_centro,
                                                     table=self.table_mean, th_failure_extrapolate=EpisodesParamsDouble.TH_FAILURE_EXTRAPOLATE)




        success_ep2, failure_ep2, valleys_rackets1, valleys_rackets2, valleys_wall, valleys_table = self.findEpisodesDouble(
            clean_ball,
            smooth_r1_data,
            smooth_r2_data,
            wall=self.wall_centro,
            table=self.table_mean
            ,params=EpisodesParamsDouble("clean_ball"))

        print("Before cleaning")
        print("Success: " + str(len(success_ep)))
        print("Failure: " + str(len(failure_ep)))

        print("After cleaning")
        print("Success: " + str(len(success_ep2)))
        print("Failure: " + str(len(failure_ep2)))

        plt.plot(success_ep[:, 0], np.repeat(20, success_ep.shape[0]), label="peaks", color="green", marker="o",
                 linestyle="None", alpha=0.5)
        plt.plot(success_ep[:, 1], np.repeat(20, success_ep.shape[0]), label="peaks", color="green", marker="o",
                 linestyle="None", alpha=0.5)

        plt.plot(success_ep2[:, 0], np.repeat(20, success_ep2.shape[0]), label="peaks", color="blue", marker="o",
                 linestyle="None", alpha=0.5)
        plt.plot(success_ep2[:, 1], np.repeat(20, success_ep2.shape[0]), label="peaks", color="blue", marker="o",
                 linestyle="None", alpha=0.5)
        plt.show()

        return clean_ball

if __name__ == '__main__':
    reader = ViconReader()
    # obj, sub = reader.extractData("F:\\users\\prasetia\\data\\TableTennis\\Test\\2022.11.30\\T02.csv")
    # obj, sub = reader.extractData("F:\\users\\prasetia\\data\\TableTennis\\Test\\2022.12.01\\T02.csv")
    obj, sub = reader.extractData("F:\\users\\prasetia\\data\\TableTennis\\Test\\2022.11.08\\T02.csv", cleaning=True)
    # obj, sub = reader.extractData("F:\\users\\prasetia\\data\\TableTennis\\Test\\2023.02.15\\T03.csv")

    reader = BallFinding(obj, sub)
    # data = reader.cleanDoubleData("F:\\users\\prasetia\\data\\TableTennis\\Test\\2022.11.30\\T02.c3d")
    # data = reader.cleanDoubleData("F:\\users\\prasetia\\data\\TableTennis\\Test\\2022.12.01\\T02.c3d")
    data = reader.cleanDoubleData("F:\\users\\prasetia\\data\\TableTennis\\Test\\2022.11.08\\T02.c3d")
    # data = reader.cleanDoubleData("F:\\users\\prasetia\\data\\TableTennis\\Test\\2023.02.15\\T03.c3d")

    df = pd.DataFrame(data, columns=["ball_x", "ball_y", "ball_z"])
    df.to_csv("F:\\users\\prasetia\\data\\TableTennis\\Test\\2022.11.08\\T02_ball.csv")
    print(len(data))
