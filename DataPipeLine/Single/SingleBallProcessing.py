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
from BallProcessing import BallProcessing
class SingleBallCleaning(BallProcessing):


    def __init__(self, obj: list, sub: list, session_name: str):
        super().__init__(obj, sub, session_name)
    def cleanSingleData(self, file_path: str = None):
        data = ezc3d.c3d(file_path)
        labels = data['parameters']['POINT']['LABELS']['value']
        unlabeled_idx = [i for i in range(len(labels)) if
                         "*" in labels[i]]  # the column label of the unlabelled marker starts with *
        data_points = np.array(data['data']['points'])

        unlabelled_data = data_points[0:3, unlabeled_idx, :]
        tobii_data = []
        for s in self.subjects:
            tobii_data.append(s["segments"].filter(regex='TobiiGlass_T').values)  # tobii data are the 3 last columns
        tobii_data = np.array(tobii_data)
        normalized_data = self.filteringUnLabData(unlabelled_data, tobii_data)
        smooth_ball_data = np.array([movingAverage(normalized_data[:, i], n=1) for i in range(3)]).transpose()
        smooth_r1_data = np.array(
            [movingAverage(self.racket_1["segments"].filter(regex='pt_T').values[:, i], n=1) for i in
             range(3)]).transpose()

        success_ep, failure_ep, valleys_rackets, valleys_wall, valleys_table = self.findEpisodesSingle(smooth_ball_data,
                                                                                                       smooth_r1_data,
                                                                                                       wall=self.wall_centro,
                                                                                                       table=self.table_mean
                                                                                                       ,
                                                                                                       params=EpisodesParamsSingle(
                                                                                                           "not_clean"),
                                                                                                       show=True)
        clean_ball = self.extrapolateInterpolateBall(smooth_ball_data, success_ep, failure_ep, valleys_wall,
                                                     valleys_table,
                                                     wall=self.wall_centro,
                                                     table=self.table_mean,
                                                     th_failure_extrapolate=EpisodesParamsSingle.TH_FAILURE_EXTRAPOLATE)

        success_ep2, failure_ep2, valleys_rackets2, valleys_wall2, valleys_table2 = self.findEpisodesSingle(clean_ball,
                                                                                                            smooth_r1_data,
                                                                                                            wall=self.wall_centro,
                                                                                                            table=self.table_mean,
                                                                                                            params=EpisodesParamsSingle(
                                                                                                                "clean_ball"),
                                                                                                            show=True)
        # print("Before cleaning")
        # print("Success: " + str(len(success_ep)))
        # print("Failure: " + str(len(failure_ep)))
        #
        # print("After cleaning")
        # print("Success: " + str(len(success_ep2)))
        # print("Failure: " + str(len(failure_ep2)))

        print("%s, %d, %d, %d, %d" % (
            self.session_name, len(success_ep), len(failure_ep), len(success_ep2), len(failure_ep2)))

        return clean_ball, self.contructValleyWallTable(success_ep2, valleys_wall2,
                                                        valleys_table2), self.constructFailureEpisodes(success_ep2,
                                                                                                       failure_ep2,
                                                                                                       valleys_wall2,
                                                                                                       valleys_table2)

    def findEpisodesSingle(self, ball, r1, wall=None, table=None, params: EpisodesParamsSingle = None, show=False):

        '''
        :param ball: ball trajectory
        :param r1: racket 1 trajectory
        :param r2: racket 2 trajectory
        :return:
        '''

        def groupEpisodes(idx, wall_vy=None, table_vy=None, th=150, th_failure=400, th_failure_sanity=100,
                          th_success=250):
            # check whether the ball inside the table or not
            inside_outside_table = Delaunay(self.table_area).find_simplex(ball) >= 0
            valleys_table_outside = valleys_table[(inside_outside_table[valleys_table.astype(int)] == False)]
            sucess_idx = []
            failure_idx = []
            i = 0
            while i < (len(idx) - 1):
                # print(idx[i])
                # 2023-01-16_A_T05
                # if (idx[i] == 22414.0) | (idx[i] == 22415.0):
                #     sucess_idx.append([idx[i], idx[i + 1]])
                # else:

                check_wall_valley = (wall_vy > idx[i]) & (wall_vy < idx[i + 1])

                if (idx[i + 1] - idx[i] < th) & np.sum(check_wall_valley) > 0:
                    curr_wall = wall_vy[((wall_vy > idx[i]) & (wall_vy < idx[i + 1]))][-1]
                    table_in_episode = (table_vy > curr_wall) & (table_vy < idx[i + 1])
                    table_all_episode = (table_vy > idx[i]) & (table_vy < idx[i + 1])
                    check_table_valley = np.sum(table_in_episode) == 1
                    check_table_all_ep = np.sum(table_all_episode) == 1
                    if check_table_valley:
                        if check_table_all_ep:
                            sucess_idx.append([idx[i], idx[i + 1]])
                        else:
                            if i > 0:
                                if idx[i] - idx[i - 1] < th_success:
                                    failure_idx.append(idx[i])
                                else:
                                    sucess_idx.append([idx[i], idx[i + 1]])
                            else:
                                sucess_idx.append([idx[i], idx[i + 1]])

                    else:
                        if (len(table_vy[table_in_episode == True]) == 0):
                            failure_idx.append(idx[i])
                        else:
                            table_last = table_vy[table_in_episode == True][-1]

                            # double table valley (one inside table and the other is outside)
                            # (idx[i+1] - table_last < 5) the ball almost reach table and the individual succeed to respond
                            if (np.isin(table_last, valleys_table_outside) | (idx[i + 1] - table_last < 5)) & (
                                    np.sum(table_in_episode) == 2):
                                sucess_idx.append([idx[i], idx[i + 1]])
                            else:
                                # print("failure")
                                failure_idx.append(idx[i])
                    # i+=1
                else:
                    # print(idx[i])
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

        # ball[12500:12550] = np.nan
        dist_rackets = interPolateDistance(np.linalg.norm(ball - r1, axis=-1))
        dist_walll = interPolateDistance(np.abs(ball[:, 1] - wall[1]))
        dist_table = interPolateDistance(np.abs(ball[:, 2] - (table[2])))
        # get valleys wall
        valleys_wall = findValleys(dist_walll, th_c=params.TH_CONFIDENCE, th_d=params.TH_D_WALL)
        valleys_wall = groupValleys(valleys_wall, dist_walll, within_th=params.TH_WITHIN, n_group=(1, 50))

        # get valleys racket 1
        valleys_rackets = findValleys(dist_rackets, th_c=params.TH_CONFIDENCE, th_d=params.TH_D_RACKET)
        valleys_rackets = groupValleys(valleys_rackets, dist_rackets, within_th=params.TH_WITHIN_RACKET,
                                       n_group=(1, 150))

        # get valleys table
        valleys_table = findValleys(dist_table, th_c=params.TH_CONFIDENCE, th_d=params.TH_D_TABLE)
        # some people hit the ball when it is near the table, remove the valley before impact
        valleys_table = removeSpecialValleyTable(valleys_table, valleys_rackets)
        valleys_table = groupValleys(valleys_table, dist_table, within_th=params.TH_WITHIN, n_group=(1, 50))

        # check valley sanity
        valleys_rackets = checkValleysSanity(valleys_rackets, valleys_wall)

        # delete idx
        # valleys_rackets = np.delete(valleys_rackets, np.argwhere((valleys_rackets == 11905)|(valleys_rackets == 15467)|(valleys_rackets == 16840)|(valleys_rackets == 18402)|(valleys_rackets == 19726)))
        # valleys_table = np.delete(valleys_table, np.argwhere((valleys_table == 2258 )| (valleys_table == 3369)))
        # valleys_table = np.delete(valleys_table, np.argwhere((valleys_table == 14407) | (valleys_table == 20641)))
        # valleys_rackets = np.delete(valleys_rackets, np.argwhere((valleys_rackets == 6877)))
        success_ep, failure_ep = groupEpisodes(valleys_rackets, valleys_wall, valleys_table,
                                               th=params.TH_SUCCESS_EPISODES,
                                               th_failure_sanity=params.TH_FAILURE_SANITY,
                                               th_failure=params.TH_FAILURE_MID_EPISODES,
                                               th_success=params.TH_SUCCESS_EPISODES)
        failure_ep = np.sort(failure_ep)

        if show:
            import matplotlib.pyplot as plt
            plt.plot(np.arange(len(dist_rackets)), dist_rackets, label="dist", color="#66c2a5", linewidth=1)
            plt.plot(np.arange(len(dist_walll)), dist_walll, label="dist wall", color="#8da0cb", linewidth=1)
            plt.plot(np.arange(len(dist_table)), dist_table, label="dist wall", color="#e78ac3", linewidth=1)

            plt.plot(valleys_table, np.repeat(70, valleys_table.shape[0]), label="peaks", color="black", marker="o",
                     linestyle="None", alpha=0.5)
            plt.plot(valleys_rackets, np.repeat(70, valleys_rackets.shape[0]), label="peaks", color="yellow",
                     marker="o",
                     linestyle="None", alpha=0.5)
            plt.plot(success_ep[:, 0], np.repeat(70, success_ep.shape[0]), label="peaks", color="green", marker="o",
                     linestyle="None", alpha=0.5)
            plt.plot(success_ep[:, 1], np.repeat(70, success_ep.shape[0]), label="peaks", color="green", marker="o",
                     linestyle="None", alpha=0.5)
            plt.plot(failure_ep, np.repeat(70, failure_ep.shape[0]), label="peaks", color="red", marker="o",
                     linestyle="None", alpha=0.5)
            plt.plot(valleys_wall, np.repeat(70, valleys_wall.shape[0]), label="peaks", color="blue", marker="o",
                     linestyle="None", alpha=0.5)
            plt.show()

        return success_ep, failure_ep, valleys_rackets, valleys_wall, valleys_table

    def constructFailureEpisodes(self, success, failures, wall, table):
        success_end = success[:, 1]
        success_start = success[:, 0]
        failures_episodes = []
        for f in failures:
            fs = success_start[success_end == f]
            if len(fs) > 0:
                wall_i = wall[(wall > fs[0]) & (wall < f)]
                table_i = table[(table > fs[0]) & (table < f)]
                failures_episodes.append([fs[0], f, wall_i[0], table_i[-1]])

        return np.asarray(failures_episodes, dtype=int)

    def contructValleyWallTable(self, success, wall, table):
        wall_idx = []
        table_idx = []

        for s in success:
            wall_i = wall[(wall > s[0]) & (wall < s[1])]
            table_i = table[(table > s[0]) & (table < s[1])]

            if len(table_i) == 2:
                table_i = table_i[-1:]
            wall_idx.append(wall_i)
            table_idx.append(table_i)

        return np.concatenate([success, wall_idx, table_idx], axis=1).astype(int)