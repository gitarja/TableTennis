import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Utils.Lib import wienerFilter, movingAverage, savgolFilter, cartesianToPolar
from GazeEvent import detectALPhase1, detectALPhase2
import pandas as pd
sns.despine(offset=10, trim=True, left=True)
sns.set_palette(sns.color_palette("Set2"))
from Utils.DataReader import TobiiReader
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


class Classic:

    def __init__(self, sub, racket, ball, tobii, table, wall):
        # tobii reader
        self.tobii_reader = TobiiReader()

        s = sub["segments"]
        r = racket["segments"]
        tobii = tobii["trajectories"]

        self.eye_event = tobii.loc[:]["Eye_movement_type"].values.astype(float)
        self.tobii_time = tobii.loc[:]["Timestamp"].values.astype(float)
        tobii.iloc[tobii["Timestamp"] == 0, 1:] = np.nan
        self.gaze_point = tobii.filter(regex='Gaze_point_3D').values
        self.left_gaze_dir = tobii.filter(regex='Gaze_direction_left').values
        self.left_eye = tobii.filter(regex='Pupil_position_left').values
        self.right_eye = tobii.filter(regex='Pupil_position_right').values
        self.right_gaze_dir = tobii.filter(regex='Gaze_direction_right').values

        self.success_idx = ball["success_idx"]
        self.failures_idx = ball["failures_idx"]

        # trajectories
        self.wall_T = wall["trajectories"].values.reshape((-1, 4, 3))
        self.table_T = table["trajectories"].values.reshape((-1, 4, 3))
        table = table["trajectories"].values.reshape((-1, 3))

        # remove failure episodes from success
        self.success_idx = self.success_idx[~np.in1d(self.success_idx[:, 1], self.failures_idx[:, 1])]

        # Get previous episodes
        self.prev_s_idx = np.argwhere(np.isin(self.success_idx[:, 1], self.success_idx[:, 0]))
        self.prev_f_idx = np.argwhere(np.isin(self.success_idx[:, 1], self.failures_idx[:, 0]))
        # get segments
        self.table_segment = np.nanmean(table, axis=0, keepdims=True)
        self.root_segment_T = s.filter(regex='Root_T').values
        self.rwirst_segment_T = s.filter(regex='R_Wrist_T').values
        self.rhummer_segment_T = s.filter(regex='R_Humerus_T').values
        self.relbow_segment_T = s.filter(regex='R_Elbow_T').values

        self.racket_segment_T = r.filter(regex='pt_T').values
        self.racket_segment_R = r.filter(regex='pt_R').values

        self.tobii_segment_T = s.filter(regex='TobiiGlass_T').values
        self.tobii_segment_R = s.filter(regex='TobiiGlass_R').values

        # get trajectories
        self.ball_t = self.smoothBall(ball["trajectories"].values, self.success_idx, self.failures_idx)

        # normalize tobii dta
        self.normalizeTobii()

    def normalizeTobii(self):
        for e in np.vstack([self.success_idx, self.failures_idx]):
            start = e[0]
            stop = e[1]

            self.left_gaze_dir[start:stop] = self.tobii_reader.gapFill(self.left_gaze_dir[start:stop])
            self.right_gaze_dir[start:stop] = self.tobii_reader.gapFill(self.right_gaze_dir[start:stop])
            self.gaze_point[start:stop] = self.tobii_reader.gapFill(self.gaze_point[start:stop])

    def smoothBall(self, ball:np.array) -> np.array:
        ball = np.array([movingAverage(ball[:, i], n=1) for i in range(3)]).transpose()
        return ball

    def eventToLabels(self, e_idx:int=0):
        if e_idx==0:
            return "Ball_Hit"
        elif e_idx==2:
            return "Bounce_Wall"
        elif e_idx==3:
            return "Bounce_Table"
        elif e_idx==1:
            return "Ball_Impact"
        else:
            return "Unknown"


    def distanceBallBeforeEvent(self, ball:np.array, episodes: np.array, segment: np.array, e_idx: int= 1, n:int =10):
        '''
        :param ball: ball trajectory vector
        :param episodes: episodes
        :param segment: segment trajectory vector
        :param e_idx: index of event
        :param n: n frames before event
        :return:
        - event name
        - the distance before time
        - the distance for each axis
        '''
        segment_bounce = segment[episodes[:, e_idx] - n]  # position of the segment when the ball bounces on the table
        segment_end = ball[episodes[:, e_idx]]  # position of the last ball

        raw_axis_dist = np.sqrt(np.square(segment[episodes[:, e_idx] - n] - ball[episodes[:, e_idx]]))
        d_bfe = np.linalg.norm(segment_end - segment_bounce, axis=-1)

        features_summary = {"event": self.eventToLabels(e_idx),
         "d_before_event": d_bfe,
         "d_axis": raw_axis_dist}

        return features_summary

    def distanceBallOverTime(self, ball:np.array, segment, episode, e_idx:int=1, n:int=15):
        '''
        :param ball: ball trajectory vector
        :param episodes: episodes
        :param segment: segment trajectory vector
        :param e_idx: index of event
        :param n: n frames before event
        :return:
         - event name
         - distance in phase
        '''
        dists = []
        for e in episode:
            start = e[e_idx] - n
            stop = e[e_idx]

            segment_t = segment[start:stop]
            ball_t = ball[start:stop]
            dist = np.linalg.norm(ball_t - segment_t, axis=-1)
            dists.append(dist)
        return np.asarray(dists, dtype=float)


    def computeSaccadePursuit(self, episode, th_al_p1=10, th_al_p2=25, th_angle_p=15):
        '''
        Detect the onset and offset of saccade and pursuit in 3 phases
        :param th_al_p1:  degree/frame (saccade offset - ball trajectory) of AL in phase 1
        :param th_al_p2:  degree/frame (saccade offset - ball trajectory) of AL in phase 2
        :param th_angle_p:   degree/frame (saccade offset - ball trajectory) of pursuit in phase 3
        :return:
        % pursuit in phase 3
        % AL in phase 1 and 2
        % onset and offset AL in phase 1 and the min angle
        % onset and offset AL in phase 2 and the angle (saccade_offset_gaze - pursuit_onset_ball, azimuth only)
        % onset and offset pursuit in phase 3 and avg angle
        '''

        ball_n = self.tobii_reader.global2LocalGaze(self.ball_t, self.tobii_segment_T, self.tobii_segment_R,
                                                    translation=True)

        gaze_n = self.gaze_point

        _, b_az, b_elv = cartesianToPolar(ball_n, swap=True)
        _, g_az, g_elv = cartesianToPolar(gaze_n, swap=True)

        az = g_az - b_az
        elv = g_elv - b_elv
        gaze_an = np.vstack([g_az, g_elv]).T
        ball_an = np.vstack([b_az, b_elv]).T

        p1_features_list = []
        p2_features_list = []
        p3_features_list = []
        for se in episode:
            # phase 1
            p1_s = se[0]
            p1_e = se[2]

            # phase 2
            p2_s = se[2]
            p2_e = se[3]

            # phase 3
            p3_s = se[3]
            p3_e = se[1]

            phase1_features = detectALPhase1(self.eye_event[p1_s:p1_e], az[p1_s:p1_e], elv[p1_s:p1_e], th=th_al_p1)
            phase2_features, phase3_features = detectALPhase2(self.eye_event[p2_s:p2_e], self.eye_event[p3_s:p3_e],
                                                              gaze_an[p2_s:p3_e], ball_an[p2_s:p3_e], th=th_al_p2,
                                                              th_angle_p=th_angle_p)

            p1_features_list.append(phase1_features)
            p2_features_list.append(phase2_features)
            p3_features_list.append(phase3_features)

        p1_features_list = np.vstack(p1_features_list)
        p2_features_list = np.vstack(p2_features_list)
        p3_features_list = np.vstack(p3_features_list)
        features_summary = {
            "al_p1_percentage": np.average(p1_features_list[:, 0] != 1e+4),
            "al_p2_percentage": np.average(p2_features_list[:, 0] != 1e+4),
            "pr_p2_percentage": np.average(p3_features_list[:, 0] != 1e+4),
            "p1_features": p1_features_list[p1_features_list[:, 0] != 1e+4],
            "p2_features": p2_features_list[p2_features_list[:, 0] != 1e+4],
            "p3_features": p3_features_list[p3_features_list[:, 0] != 1e+4]
        }

        return features_summary

    def computeGazeBallAngle(self, n=20, relative=False, episodes=None):
        ball = self.ball_t
        tobii = self.tobii_segment_T

        l_eye_gd = self.left_gaze_dir
        r_eye_gd = self.right_gaze_dir

        ball_n = ball - tobii
        # ball_n = ball_n[:, [0, 2]]
        ball_n = ball_n / np.linalg.norm(ball_n, axis=-1, keepdims=True)

        l_gaze_n = l_eye_gd - tobii
        r_gaze_n = r_eye_gd - tobii
        gaze_n = (l_gaze_n + r_gaze_n) / 2
        # gaze_n = gaze_n[:, [0, 2]]
        gaze_n = gaze_n / np.linalg.norm(gaze_n, axis=-1, keepdims=True)

        # gaze_n = self.gaze_point - tobii
        # gaze_n = gaze_n / np.linalg.norm(gaze_n, axis=-1, keepdims=True)

        if relative:
            def extract(episodes):
                angle_list = []
                for e in episodes:
                    ball_nt = ball_n[e[2]]
                    gaze_nt = gaze_n[e[2] - n:e[2] + n]
                    aug_ball_nt = np.ones_like(gaze_nt) * ball_nt
                    angle = np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i', aug_ball_nt, gaze_nt), -1.0, 1.0)))
                    speed_angel = np.sqrt(np.square(angle[1:] - angle[:-1]))
                    if (np.sum(np.isnan(angle)) == 0):
                        angle_list.append(angle)

                    # plt.plot(angle)
                    #
                    # plt.show()

                return np.asarray(angle_list, dtype=float)

        else:
            angles = np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i', ball_n, gaze_n), -1.0, 1.0)))

            def extract(episodes):
                angle_list = []
                for e in episodes:
                    # print(e)
                    start = e[2] - n
                    stop = e[2] + n
                    angles_t = angles[start: stop]

                    idx_e = 2
                    # print(angles[e[idx_e]])
                    wall_n = self.wall_T[e[idx_e]] - self.tobii_segment_T[e[idx_e]]
                    wall_n = wall_n / np.linalg.norm(wall_n, axis=-1, keepdims=True)

                    table_n = self.table_T[e[idx_e]] - self.tobii_segment_T[e[idx_e]]
                    table_n = table_n / np.linalg.norm(table_n, axis=-1, keepdims=True)


                    if np.sum(np.isnan(angles_t)) == 0:
                        angle_list.append(angles_t)
                return np.asarray(angle_list, dtype=float)

        return extract(episodes)

    def extractGazeBallPolar(self, n=50, episode_idx=2, n_group=2):

        az_inc_success = self.computeSaccadePursuit(self.success_idx)
        # az_inc_failures = self.computeGazeBallPolar(n, episode_idx, episodes=self.failures_idx, n_group=n_group)

        return az_inc_success, az_inc_success

    def extractAnglesBeforeImpact(self, n=50):

        rwirst = self.ball_t
        rhummer = self.racket_segment

        angles = np.rad2deg(self.computeSegmentAngles(rwirst, rhummer))

        def extract(episodes):
            angle_list = []
            for e in episodes:
                start = e[1] - 5
                stop = e[1] + 1
                angles_t = angles[start: stop]
                angle_list.append(angles_t)
            return np.asarray(angle_list, dtype=float)

        angles_success = extract(self.success_idx)
        angles_fail = extract(self.failures_idx)

        return angles_success, angles_fail

    def extractDistanceBeforeImpact(self, n=15):
        dist_success = self.computeDistanceOverTime(self.ball_t, self.racket_segment, self.success_idx, n=n)
        dist_fail = self.computeDistanceOverTime(self.ball_t, self.racket_segment, self.failures_idx, n=n)

        return np.diff(dist_success, n=0, axis=-1), np.diff(dist_fail, n=0, axis=-1)

    def extractDistanceRacketBeforeImpact(self, n=15):
        dist = np.linalg.norm(self.table_segment[:, 1:] - self.racket_segment[:, 1:], axis=-1)

        def extract(episodes):
            dist_list = []
            rt_list = []
            for e in episodes:
                start = e[1] - n
                stop = e[1] + 1
                angles_t = dist[start: stop]

                rt_dist = dist[e[0]: e[1]]
                peaks, _ = find_peaks(rt_dist, distance=50)

                if len(peaks) == 0:
                    print(e)
                    # plt.plot(rt_dist)
                    # plt.plot(peaks, rt_dist[peaks], "x")
                    # plt.show()
                else:
                    # print(e)
                    # plt.plot(rt_dist)
                    # plt.plot(peaks, rt_dist[peaks], "x")
                    # plt.show()
                    rt = (e[1] - e[0]) - peaks[-1]
                    rt_list.append(rt)

                dist_list.append(angles_t)

            # print(rt_list)
            return np.asarray(dist_list, dtype=float), np.asarray(rt_list, dtype=float)

        dist_success, rt_success = extract(self.success_idx)
        dist_fail, rt_failure = extract(self.failures_idx)

        return dist_success, dist_fail, rt_success, rt_failure

    def extractBouncePoints(self):
        s_wall, s_table = self.computeHistBouce(self.ball_t, self.success_idx)
        f_wall, f_table = self.computeHistBouce(self.ball_t, self.failures_idx)

        return s_wall, s_table, f_wall, f_table

    def extractAnglePersonBall(self):
        s_angles = self.computeAnglePersonBall(self.ball_t, self.success_idx, self.racket_segment)
        f_angles = self.computeAnglePersonBall(self.ball_t, self.failures_idx, self.racket_segment)

        return s_angles, f_angles

    def extractDistanceBounceEnd(self):
        s_dist, s_dist_raw = self.computeDistanceBounce(self.ball_t, self.success_idx, self.racket_segment)
        f_dist, f_dist_raw = self.computeDistanceBounce(self.ball_t, self.failures_idx, self.racket_segment)

        return s_dist, f_dist, s_dist_raw, f_dist_raw

    def extractGazeBallAngle(self, n=20, relative=False):

        angles_success = self.computeGazeBallAngle(n, relative, self.success_idx)
        angles_fail = self.computeGazeBallAngle(n, relative, self.failures_idx)

        return angles_success, angles_fail

    def extractVelocityBallRacket(self):
        s_b, v_b, a_b = self.computeVelAcc(self.ball_t)
        s_r, v_r, a_r = self.computeVelAcc(self.racket_segment)

        def computeVelocityBeforeImpact(episodes):
            vr_list = []
            vb_list = []
            vr_all = []
            for e in episodes:
                start = e[1] - 5
                stop = e[1] - 2
                # vr_list.append(np.max(a_b[start:e[1]+1]))
                vr_all.append(a_r[stop - 15: stop])
                vr_list.append(np.average(a_r[start: stop]))
                vb_list.append(np.average(a_b[start: stop]))

            return np.asarray(vb_list, dtype=float), np.asarray(vr_list, dtype=float), np.asarray(vr_all, dtype=float)

        vs_b, vs_r, vs_rall = computeVelocityBeforeImpact(self.success_idx)
        vf_b, vf_r, vf_rall = computeVelocityBeforeImpact(self.failures_idx)

        return vs_b, vs_r, vf_b, vf_r, vs_rall, vf_rall

    def extractVarinceMovementBI(self):

        def computeEpisodes(episodes, segment):
            s_var = []
            for e in episodes:
                start = e[3]
                stop = e[1]
                # s_var.append(np.std(segment[start:stop, 0:1]))
                s_var.append(np.sum(np.std(segment[start:stop, 1:], axis=0)))
            return np.asarray(s_var, dtype=float)

        s_var = computeEpisodes(self.success_idx, self.rwirst_segment_T)
        f_var = computeEpisodes(self.failures_idx, self.rwirst_segment_T)

        return s_var, f_var

    def extractEvents(self):
        bounce_table_s = self.success_idx[:, 1] - self.success_idx[:, 2]
        bounce_wall_s = self.success_idx[:, 1] - self.success_idx[:, 3]

        bounce_table_f = self.failures_idx[:, 1] - self.failures_idx[:, 2]
        bounce_wall_f = self.failures_idx[:, 1] - self.failures_idx[:, 3]

        return bounce_table_s, bounce_wall_s, bounce_table_f, bounce_wall_f

    def extractImpactPositions(self):
        impact_s = self.ball_t[self.success_idx[:, 1]]
        impact_f = self.ball_t[self.failures_idx[:, 1]]

        return impact_s, impact_f

    def extractPrevCurrentImpactPositions(self):
        def extract(ref_episodes, episodes, idx):
            dist_list = []
            for i in idx:
                prev_e = ref_episodes[i][0]
                curr_e = episodes[episodes[:, 0] == prev_e[1]][0]
                prev_impact = self.ball_t[prev_e[1]]
                curr_impact = self.ball_t[curr_e[1]]

                dist = np.sqrt(np.square(prev_impact - curr_impact))
                dist_list.append(dist)

            return np.asarray(dist_list)

        dist_pc_s = extract(self.success_idx, self.success_idx, self.prev_s_idx)
        dist_pc_f = extract(self.success_idx, self.failures_idx, self.prev_f_idx)

        return dist_pc_s, dist_pc_f

    def extractPrevVelocityBallRacket(self):
        s_b, v_b, a_b = self.computeVelAcc(self.ball_t)
        s_r, v_r, a_r = self.computeVelAcc(self.racket_segment)

        def computeVelocityBeforeImpact(episodes):
            vr_list = []
            vb_list = []
            vr_all = []
            for e in episodes:
                start = e[1] - 5
                stop = e[1] - 2
                # vr_list.append(np.max(a_b[start:e[1]+1]))
                vr_all.append(a_r[stop - 15: stop])
                vr_list.append(np.average(a_r[start: stop]))
                vb_list.append(np.average(a_b[start: stop]))

            return np.asarray(vb_list, dtype=float), np.asarray(vr_list, dtype=float), np.asarray(vr_all, dtype=float)

        vs_b, vs_r, vs_rall = computeVelocityBeforeImpact(self.success_idx[self.prev_s_idx[:, 0]])
        vf_b, vf_r, vf_rall = computeVelocityBeforeImpact(self.success_idx[self.prev_f_idx[:, 0]])

        return vs_b, vs_r, vf_b, vf_r, vs_rall, vf_rall

    def extractPrevDistanceRacketBeforeImpact(self, n=15):
        dist = np.linalg.norm(self.table_segment[:, 1:] - self.racket_segment[:, 1:], axis=-1)

        def extract(episodes):
            dist_list = []
            rt_list = []
            for e in episodes:
                start = e[1] - n
                stop = e[1] + 1
                angles_t = dist[start: stop]
                rt_dist = dist[e[0]: e[1]]
                peaks, _ = find_peaks(rt_dist, distance=50)

                if len(peaks) == 0:
                    print(e)
                else:

                    rt = (e[1] - e[0]) - peaks[-1]
                    rt_list.append(rt)
                dist_list.append(angles_t)

            return np.asarray(dist_list, dtype=float), np.asarray(rt_list, dtype=float)

        dist_success, rt_success = extract(self.success_idx[self.prev_s_idx[:, 0]])
        dist_fail, rt_failure = extract(self.success_idx[self.prev_f_idx[:, 0]])

        return dist_success, dist_fail, rt_success, rt_failure

    def extractPrevGazeBallAngle(self, n=20, relative=False):

        angles_success = self.computeGazeBallAngle(n, relative, self.success_idx[self.prev_s_idx[:, 0]])
        angles_fail = self.computeGazeBallAngle(n, relative, self.success_idx[self.prev_f_idx[:, 0]])

        return angles_success, angles_fail






if __name__ == '__main__':
    from Utils.DataReader import SubjectObjectReader

    reader = SubjectObjectReader()
    visual = Visualization()
    paths = [
        "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-08_A\\2022-11-08_A_T01_complete.pkl",
        "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-08_A\\2022-11-08_A_T03_complete.pkl",
        "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-08_A\\2022-11-08_A_T04_complete.pkl",
        "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-09_A\\2022-11-09_A_T07_complete.pkl",
        "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-09_A\\2022-11-09_A_T04_complete.pkl",
        "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-09_A\\2022-11-09_A_T03_complete.pkl",
        "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2023-02-08_A\\2023-02-08_A_T02_complete.pkl",
        "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2023-02-08_A\\2023-02-08_A_T04_complete.pkl",
        "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2023-02-08_A\\2023-02-08_A_T03_complete.pkl"
    ]


