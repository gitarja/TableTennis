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



class DoulbeBallProcessing:

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
            if len(table_i) == 2:
                table_i = table_i[-1:]
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