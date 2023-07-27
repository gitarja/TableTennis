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
from PreProcessing.BallProcessing import BallProcessing


class DoulbeBallProcessing(BallProcessing):

    def __init__(self, obj: list, sub: list, session_name: str):
        super().__init__(obj, sub, session_name, double=True)

    def contructValleyWallTableDouble(self, success, ball, r1, r2, wall, table):
        wall_idx = []
        table_idx = []
        hit1_idx = []
        hit2_idx = []

        dist_rackets1 = interPolateDistance(np.linalg.norm(ball - r1, axis=-1))
        dist_rackets1[np.isnan(np.sum(r1, 1))] = 10000
        dist_rackets2 = interPolateDistance(np.linalg.norm(ball - r2, axis=-1))
        dist_rackets2[np.isnan(np.sum(r2, 1))] = 10000
        for s in success:
            wall_i = wall[(wall > s[0]) & (wall < s[1])]
            table_i = table[(table > s[0]) & (table < s[1])]
            hit1_i = np.argmin([dist_rackets1[int(s[0])], dist_rackets2[int(s[0])]]) + 1
            hit2_i = np.argmin([dist_rackets1[int(s[1])], dist_rackets2[int(s[1])]]) + 1
            if (len(table_i) == 2) :
                table_i = table_i[-1:]
            elif (len(table_i) == 3):
                table_i = table_i[1:2]
            wall_idx.append(wall_i)
            table_idx.append(table_i)
            hit1_idx.append(np.array([hit1_i]))
            hit2_idx.append(np.array([hit2_i]))

        return np.concatenate([success, wall_idx, table_idx, hit1_idx, hit2_idx], axis=1).astype(int)

    def constructFailureEpisodesDouble(self, success, ball, r1, r2, failures, wall, table):
        success_end = success[:, 1]
        success_start = success[:, 0]
        failures_episodes = []

        dist_rackets1 = interPolateDistance(np.linalg.norm(ball - r1, axis=-1))
        dist_rackets1[np.isnan(np.sum(r1, 1))] = 10000
        dist_rackets2 = interPolateDistance(np.linalg.norm(ball - r2, axis=-1))
        dist_rackets2[np.isnan(np.sum(r2, 1))] = 10000

        for f in failures:
            fs = success_start[success_end == f]
            if len(fs) > 0:
                hit1_i = np.argmin([dist_rackets1[int(fs[0])], dist_rackets2[int(fs[0])]]) + 1
                hit2_i = np.argmin([dist_rackets1[int(f)], dist_rackets2[int(f)]]) + 1
                wall_i = wall[(wall > fs[0]) & (wall < f)]
                table_i = table[(table > fs[0]) & (table < f)]
                failures_episodes.append([fs[0], f, wall_i[0], table_i[-1], hit1_i, hit2_i])

        return np.asarray(failures_episodes, dtype=int)


    def cleanDoubleData(self, file_path: str = None):
        data = ezc3d.c3d(file_path)
        labels = data['parameters']['POINT']['LABELS']['value']
        unlabeled_idx = [i for i in range(len(labels)) if
                         "*" in labels[i]]  # the column label of the unlabelled marker starts with *
        data_points = np.array(data['data']['points'])

        unlabelled_data = data_points[0:3, unlabeled_idx, :]
        tobii_data = []
        for s in self.subjects:
            tobii_segment = s["segments"].filter(regex='TobiiGlass_T').values
            tobii_data.append(tobii_segment)  # tobii data are the 3 last columns

        tobii_data = np.array(tobii_data)
        normalized_data = self.filteringUnLabData(unlabelled_data, tobii_data)

        smooth_ball_data = np.array([movingAverage(normalized_data[:, i], n=1) for i in range(3)]).transpose()
        smooth_r1_data = np.array(
            [movingAverage(self.racket_1["segments"].filter(regex='pt_T').values[:, i], n=1) for i in
             range(3)]).transpose()
        smooth_r2_data = np.array(
            [movingAverage(self.racket_2["segments"].filter(regex='pt_T').values[:, i], n=1) for i in
             range(3)]).transpose()

        success_ep, failure_ep, valleys_rackets1, valleys_rackets2, valleys_wall, valleys_table = self.findEpisodesDouble(
            smooth_ball_data,
            smooth_r1_data,
            smooth_r2_data,
            wall=self.wall_centro,
            table=self.table_mean
            , params=EpisodesParamsDouble("not_clean"), show=False)

        clean_ball = self.extrapolateInterpolateBall(smooth_ball_data, success_ep, failure_ep, valleys_wall,
                                                     valleys_table,
                                                     wall=self.wall_centro,
                                                     table=self.table_mean,
                                                     th_failure_extrapolate=EpisodesParamsDouble.TH_FAILURE_EXTRAPOLATE)

        success_ep2, failure_ep2, valleys_rackets1, valleys_rackets2, valleys_wall2, valleys_table2 = self.findEpisodesDouble(
            clean_ball,
            smooth_r1_data,
            smooth_r2_data,
            wall=self.wall_centro,
            table=self.table_mean
            , params=EpisodesParamsDouble("clean_ball"), show=True)

        print("%s, %d, %d, %d, %d" % (
            self.session_name, len(success_ep), len(failure_ep), len(success_ep2), len(failure_ep2)))
        # print("Before cleaning")
        # print("Success: " + str(len(success_ep)))
        # print("Failure: " + str(len(failure_ep)))
        #
        # print("After cleaning")
        # print("Success: " + str(len(success_ep2)))
        # print("Failure: " + str(len(failure_ep2)))

        # plt.plot(success_ep[:, 0], np.repeat(20, success_ep.shape[0]), label="peaks", color="green", marker="o",
        #          linestyle="None", alpha=0.5)
        # plt.plot(success_ep[:, 1], np.repeat(20, success_ep.shape[0]), label="peaks", color="green", marker="o",
        #          linestyle="None", alpha=0.5)
        #
        # plt.plot(success_ep2[:, 0], np.repeat(20, success_ep2.shape[0]), label="peaks", color="blue", marker="o",
        #          linestyle="None", alpha=0.5)
        # plt.plot(success_ep2[:, 1], np.repeat(20, success_ep2.shape[0]), label="peaks", color="blue", marker="o",
        #          linestyle="None", alpha=0.5)
        # plt.show()

        return clean_ball, self.contructValleyWallTableDouble(success_ep2,
                                                              clean_ball,
                                                              smooth_r1_data,
                                                              smooth_r2_data,
                                                              valleys_wall2,
                                                              valleys_table2), \
            self.constructFailureEpisodesDouble(
            success_ep2,
            clean_ball,
            smooth_r1_data,
            smooth_r2_data,
            failure_ep2,
            valleys_wall2,
            valleys_table2)

    def findEpisodesDouble(self, ball, r1, r2, wall=None, table=None, params: EpisodesParamsDouble = None, show=False):

        def groupEpisodes(idx1, idx2, wall_vy=None, table_vy=None, th=150, th_failure=400, th_failure_sanity=100):
            # check whether the ball inside the table or not
            inside_outside_table = Delaunay(self.table_area).find_simplex(ball) >= 0
            valleys_table_outside = valleys_table[(inside_outside_table[valleys_table.astype(int)] == False)]
            sucess_idx = []
            failure_idx = []
            i = 0
            idx = np.sort(np.concatenate([idx1, idx2]))
            while i < len(idx)-1:
                # print(str(idx[i]) + " " + str(idx[i + 1]))
                check_wall_valley = (wall_vy > idx[i]) & (wall_vy < idx[i + 1])
                # check whether two valleys belong to the same person or not
                check_diff_sub = not (np.isin(idx[i], idx1) & np.isin(idx[i + 1], idx1)) | (
                        np.isin(idx[i], idx2) & np.isin(idx[i + 1],
                                                        idx2))  # check whether the valley come from different subjects
                if (idx[i + 1] - idx[i] < th) & (np.sum(check_wall_valley) > 0) & check_diff_sub:
                    curr_wall = wall_vy[((wall_vy > idx[i]) & (wall_vy < idx[i + 1]))][-1]
                    # if curr_wall == 5334:
                    #     print("Test")
                    table_in_episode = (table_vy > curr_wall) & (table_vy < idx[i + 1])
                    check_table_valley = np.sum(table_in_episode) == 1
                    # there must be one valley table between valley wall and the next valley racket
                    if check_table_valley:
                        sucess_idx.append([idx[i], idx[i + 1]])
                    else:
                        if np.sum(table_in_episode) > 1:
                            table_last = table_vy[table_in_episode == True][1]
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
        dist_table = interPolateDistance(np.abs(ball[:, 2] - (table[2])))

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

        # valleys_rackets1 = np.delete(valleys_rackets1, np.argwhere((valleys_rackets1 >= 26400) & (valleys_rackets1 <= 26550)))

        # valleys_rackets1 = np.delete(valleys_rackets1,
        #                              np.argwhere((valleys_rackets1 >= 3000) & (valleys_rackets1 <= 3070)))
        valleys_rackets1 = np.delete(valleys_rackets1,
                                     np.argwhere((valleys_rackets1 >= 26900) & (valleys_rackets1 <= 27100)))
        valleys_rackets1 = np.delete(valleys_rackets1,
                                     np.argwhere((valleys_rackets1 >= 27700) & (valleys_rackets1 <= 27800)))


        valleys_rackets1 = groupValleys(valleys_rackets1, dist_rackets1, within_th=params.TH_WITHIN_RACKET)

        # get valleys racket 2
        valleys_rackets2 = findValleys(dist_rackets2, th_c=params.TH_CONFIDENCE, th_d=params.TH_D_RACKET)
        valleys_rackets2 = np.delete(valleys_rackets2, np.argwhere((valleys_rackets2 >= 8000) & (valleys_rackets2 <= 8250)))
        # valleys_rackets2 = np.delete(valleys_rackets2,
        #                              np.argwhere((valleys_rackets2 >= 27400) & (valleys_rackets2 <= 27480)))


        # valleys_rackets2 = np.delete(valleys_rackets2, np.argwhere((valleys_rackets2 >= 21100) & (valleys_rackets2 <= 21250)))
        # valleys_rackets2 = np.delete(valleys_rackets2,
        #                              np.argwhere((valleys_rackets2 >= 23800) & (valleys_rackets2 <= 23950)))
        #
        # valleys_rackets2 = np.delete(valleys_rackets2,
        #                              np.argwhere((valleys_rackets2 >= 30675) & (valleys_rackets2 <= 31100)))
        # valleys_rackets2 = np.delete(valleys_rackets2,
        #                              np.argwhere((valleys_rackets2 >= 24800) & (valleys_rackets2 <= 32000)))

        valleys_rackets2 = groupValleys(valleys_rackets2, dist_rackets2, within_th=params.TH_WITHIN_RACKET)

        # # for 2022-12-01_A_T02
        # valleys_rackets2 = np.sort(np.append(valleys_rackets2, [30659]))

        # # for 2022-11-15_M_T02
        # valleys_rackets2 = np.sort(np.append(valleys_rackets2, [31173]))

        # # for 2022-12-05_M_T02
        # valleys_rackets2 = np.sort(np.append(valleys_rackets2, [7841]))

        # get valleys wall
        valleys_wall = findValleys(dist_walll, th_c=params.TH_CONFIDENCE, th_d=params.TH_D_WALL)
        # valleys_wall = np.delete(valleys_wall,
        #                              np.argwhere((valleys_wall >= 24800) & (valleys_wall <= 32000)))

        valleys_wall = groupValleys(valleys_wall, dist_walll, within_th=params.TH_WITHIN)
        # get valleys table
        valleys_table = findValleys(dist_table, th_c=params.TH_CONFIDENCE, th_d=params.TH_D_TABLE)
        # some people hit the ball when it is near the table, remove the valley before impact

        #
        # valleys_table = np.delete(valleys_table,
        #                              np.argwhere((valleys_table >= 24800) & (valleys_table <= 32000)))
        #
        # valleys_table = np.delete(valleys_table,
        #                                  np.argwhere((valleys_table >= 16260) & (valleys_table <= 16290)))
        #
        # valleys_table = np.delete(valleys_table,
        #                           np.argwhere((valleys_table >= 6180) & (valleys_table <= 6210)))
        # if show==False:
        #     valleys_table = np.delete(valleys_table,
        #                                  np.argwhere((valleys_table >= 23925) & (valleys_table <= 23940)))


        valleys_table = removeSpecialValleyTable(valleys_table, valleys_rackets1)
        valleys_table = removeSpecialValleyTable(valleys_table, valleys_rackets2)
        valleys_table = groupValleys(valleys_table, dist_table, within_th=params.TH_WITHIN)

        # check valley sanity
        valleys_rackets1 = checkValleysSanity(valleys_rackets1, valleys_wall, dis_th=params.TH_RACKET_SANITY)
        valleys_rackets2 = checkValleysSanity(valleys_rackets2, valleys_wall, dis_th=params.TH_RACKET_SANITY)


        # delete idx

        success_ep, failure_ep = groupEpisodes(valleys_rackets1, valleys_rackets2, valleys_wall, valleys_table,
                                               th=params.TH_SUCCESS_EPISODES,
                                               th_failure=params.TH_FAILURE_MID_EPISODES,
                                               th_failure_sanity=params.TH_FAILURE_SANITY)

        failure_ep = np.sort(failure_ep)
        plt.plot(np.arange(len(dist_rackets1)), dist_rackets1, label="dist", color="#238b45", linewidth=1)
        plt.plot(np.arange(len(dist_walll)), dist_walll, label="dist wall", color="#8da0cb", linewidth=1)
        plt.plot(np.arange(len(dist_table)), dist_table, label="dist table", color="#e78ac3", linewidth=1)
        plt.plot(np.arange(len(dist_rackets2)), dist_rackets2, label="dist", color="#66c2a4", linewidth=1)

        plt.plot(valleys_wall, np.repeat(20, valleys_wall.shape[0]), label="peaks", color="blue", marker="o",
                 linestyle="None", alpha=0.5)
        plt.plot(valleys_table, np.repeat(20, valleys_table.shape[0]), label="peaks", color="orange", marker="o",
                 linestyle="None", alpha=0.5)

        # plt.plot(valleys_rackets1, np.repeat(20, valleys_rackets1.shape[0]), label="peaks", color="black", marker="o",
        #          linestyle="None", alpha=0.5)
        # plt.plot(valleys_rackets2, np.repeat(20, valleys_rackets2.shape[0]), label="peaks", color="black", marker="o",
        #          linestyle="None", alpha=0.5)

        plt.plot(success_ep[:, 0], np.repeat(20, success_ep.shape[0]), label="peaks", color="green", marker="o",
                 linestyle="None", alpha=0.5)
        plt.plot(success_ep[:, 1], np.repeat(20, success_ep.shape[0]), label="peaks", color="green", marker="o",
                 linestyle="None", alpha=0.5)
        plt.plot(failure_ep, np.repeat(20, failure_ep.shape[0]), label="peaks", color="red", marker="o",
                 linestyle="None", alpha=0.5)

        plt.show()

        return success_ep, failure_ep, valleys_rackets1, valleys_rackets2, valleys_wall, valleys_table