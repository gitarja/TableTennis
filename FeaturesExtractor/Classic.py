import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Utils.Lib import wienerFilter, movingAverage, savgolFilter
import pandas as pd
sns.despine(offset=10, trim=True, left=True)
sns.set_palette(sns.color_palette("Set2"))
from Utils.DataReader import TobiiReader
import matplotlib.patches as mpatches
from scipy.signal import find_peaks

class Classic:

    def __init__(self, sub, racket, ball, tobii, table):
        #tobii reader
        self.tobii_reader = TobiiReader()



        s = sub["segments"]
        r = racket["segments"]
        tobii = tobii["trajectories"]
        table = table["trajectories"].values.reshape((-1, 3))
        tobii.iloc[tobii["Timestamp"] == 0, 1:] = np.nan

        self.gaze_point = tobii.filter(regex='Gaze_point_3D').values
        self.left_gaze_dir =  tobii.filter(regex='Gaze_direction_left').values
        self.left_eye =  tobii.filter(regex='Pupil_position_left').values
        self.right_eye = tobii.filter(regex='Pupil_position_right').values
        self.right_gaze_dir = tobii.filter(regex='Gaze_direction_right').values

        self.success_idx = ball["success_idx"]
        self.failures_idx = ball["failures_idx"]

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

        self.racket_segment = r.filter(regex='pt_T').values
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
            tobii_seg_T = self.tobii_segment_T[start: stop]
            tobii_seg_R = self.tobii_segment_R[start: stop]

            self.gaze_point[start:stop] = self.tobii_reader.interPolateTransform(self.gaze_point[start:stop], tobii_seg_T,
                                                                               tobii_seg_R)
            self.left_eye[start:stop] = self.tobii_reader.interPolateTransform(self.left_eye[start:stop, [0, 2, 1]], tobii_seg_T,
                                                                           tobii_seg_R)
            self.right_eye[start:stop] = self.tobii_reader.interPolateTransform(self.right_eye[start:stop, [0, 2, 1]], tobii_seg_T,
                                                                           tobii_seg_R)


            self.left_gaze_dir[start:stop] = self.tobii_reader.interPolateTransform(self.left_gaze_dir[start:stop], tobii_seg_T,
                                                                           tobii_seg_R, translation=True)
            self.right_gaze_dir[start:stop] = self.tobii_reader.interPolateTransform(self.right_gaze_dir[start:stop], tobii_seg_T,
                                                                            tobii_seg_R, translation=True)

    def computeHistBouce(self, ball, episodes):

        wall_bounce = []
        table_bounce = []
        for e in episodes:
            wall_bounce.append(ball[e[2], [0, 2]])
            table_bounce.append(ball[e[3], [0, 1]])

        return np.vstack(wall_bounce), np.vstack(table_bounce)

    def computeSegmentAngles(self, v1, v2):
        v1_u = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)  # normalize v1
        v2_u = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)  # normalize v2
        angles = np.arccos(np.clip(np.einsum('ij,ij->i', v1_u, v2_u), -1.0, 1.0))
        return angles

    def computeAnglePersonBall(self, ball, episodes, segment):

        bc = ball[episodes[:, 3], 0:3]  # bounce on the table
        be = ball[episodes[:, 1], 0:3]  # position of the last ball
        p_r = segment[episodes[:, 3], 0:3]  # segment position when the ball bounce

        angles = self.computeSegmentAngles(p_r - bc, be - bc)
        # for e in episodes:
        #     bc = ball[e[3]][1:3] # bounce on the table
        #     be = ball[e[1]][1:3] # position of the last ball
        #     p_r = person_root[e[3]][1:3]# person position when the ball bounce
        #     angles.append(self.computeSegmentAngles(p_r - bc, be - bc))

        return np.asarray(angles, dtype=float)

    def computeDistanceBounce(self, ball, episodes, segment):

        segment_bounce = segment[episodes[:, 1] - 10] # position of the segment when the ball bounces on the table
        segment_end = ball[episodes[:, 1]] # position of the last ball

        # raw_axis_dist = np.sqrt(np.square(segment[episodes[:, 3]] - ball[episodes[:, 1]]))

        raw_axis_dist = np.sqrt(np.square(segment[episodes[:, 1] - 10] - ball[episodes[:, 1]]))
        return np.linalg.norm(segment_end - segment_bounce, axis=-1), raw_axis_dist


    def computeDistanceOverTime(self, ball, segment, episode, n=15):
        dists = []

        for e in episode:
            start = e[1] - (30- 1)
            stop = e[1] + 1

            segment_t = segment[start:stop]
            ball_t = ball[stop]
            dist = np.linalg.norm(ball_t - segment_t, axis=-1)
            dists.append(dist)

        return np.asarray(dists, dtype=float)




    def computeVelAcc(self, v):

        # vel = np.sum(np.abs(np.diff(v, n=1, axis=0)), axis=-1)
        # acc = np.sum(
        #     np.diff(np.abs(np.diff(v, n=1, axis=0)), n=1, axis=0), axis=-1)

        v1 = v[:-1]
        v2 = v[1:]
        speed = np.linalg.norm(v2-v1, axis=-1)
        vel = np.sum(np.diff(v, n=1, axis=0), axis=-1)
        acc = np.diff(speed, n=1, axis=-1)

        return speed, vel, acc

    def smoothBall(self, ball, success_idx, failures_idx):

        ball = np.array([movingAverage(ball[:, i], n=1) for i in range(3)]).transpose()
        return ball

    def computeGazeBallAngle(self, n=20, relative=False, episodes=None):
        ball = self.ball_t
        tobii = self.tobii_segment_T

        l_eye_gd = self.left_gaze_dir
        r_eye_gd = self.right_gaze_dir

        ball_n = ball - tobii
        ball_n = ball_n / np.linalg.norm(ball_n, axis=-1, keepdims=True)

        l_gaze_n = l_eye_gd - tobii
        l_gaze_n = l_gaze_n / np.linalg.norm(l_gaze_n, axis=-1, keepdims=True)

        r_gaze_n = r_eye_gd - tobii
        r_gaze_n = r_gaze_n / np.linalg.norm(r_gaze_n, axis=-1, keepdims=True)

        gaze_n = (l_gaze_n + r_gaze_n) / 2

        if relative:
            def extract(episodes):
                angle_list = []
                for e in episodes:
                    ball_nt = ball_n[e[2]]
                    gaze_nt = gaze_n[e[2] - n:e[2] + n]
                    aug_ball_nt = np.ones_like(gaze_nt) * ball_nt
                    angle = np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i', aug_ball_nt, gaze_nt), -1.0, 1.0)))
                    speed_angel = np.sqrt(np.square(angle[1:] - angle[:-1]))
                    angle_list.append(angle)

                    # plt.plot(angle)
                    # plt.show()

                return np.asarray(angle_list, dtype=float)

        else:
            angles = np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i', gaze_n, ball_n), -1.0, 1.0)))

            def extract(episodes):
                angle_list = []
                for e in episodes:
                    start = e[2] - n
                    stop = e[2]  + n
                    angles_t = angles[start: stop]
                    angle_list.append(angles_t)
                return np.asarray(angle_list, dtype=float)

        return extract(episodes)


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
        dist_fail= self.computeDistanceOverTime(self.ball_t, self.racket_segment, self.failures_idx, n=n)

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
        bounce_table_s =  self.success_idx[:, 1] - self.success_idx[:, 2]
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


            return np.asarray(dist_list, dtype=float), np.asarray(rt_list, dtype=float)

        dist_success, rt_success = extract(self.success_idx[self.prev_s_idx[:, 0]])
        dist_fail, rt_failure = extract(self.success_idx[self.prev_f_idx[:, 0]])

        return dist_success, dist_fail, rt_success, rt_failure

    def extractPrevGazeBallAngle(self, n=20, relative=False):


        angles_success = self.computeGazeBallAngle(n, relative, self.success_idx[self.prev_s_idx[:, 0]])
        angles_fail = self.computeGazeBallAngle(n, relative, self.success_idx[self.prev_f_idx[:, 0]])

        return angles_success, angles_fail

class Visualization:

    def plot3D(self, success, failures):
        success_idx = np.random.choice(len(success), len(failures))
        success = success[success_idx]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(success[:, 0], success[:, 1], success[:, 2])
        ax.scatter(failures[:, 0], failures[:, 1], failures[:, 2])

        plt.show()
    def plotWallHist(self, success_bounce, fail_bounces, wall):
        success_label = ["success"] * len(success_bounce)
        fail_label = ["failures"] * len(fail_bounces)

        bouces = np.vstack([success_bounce, fail_bounces])
        labels = np.hstack([success_label, fail_label])
        df = pd.DataFrame({"x": bouces[:, 0], "y": bouces[:, 1], "label": labels})
        sns.jointplot(data=df, x="x", y="y", hue="label", type="kde")
        # gfg = sns.kdeplot(
        #     data=df, x="x", y="y", hue="label", fill=True,
        # )
        # gfg.set_ylim(0, 80)
        plt.show()

    def plotTwoHist(self, success_angle, fail_angle, y="angles(radian)"):
        np.random.seed(2023)
        success_idx = np.random.choice(len(success_angle), len(fail_angle))
        success_angle = success_angle[success_idx]
        success_label = ["success"] * len(success_angle)
        fail_label = ["failures"] * len(fail_angle)

        angles = np.hstack([success_angle, fail_angle])
        labels = np.hstack([success_label, fail_label])
        df = pd.DataFrame({y: angles, "label": labels})

        # g = sns.displot(df,  x=y, hue="label", kind="kde", fill=True)
        g = sns.catplot(df, y=y, x="label", kind="violin", split=True)

        g.set(xlabel=None)
        g.set(ylabel=None)
        # g._legend.remove()
        plt.savefig('F:\\users\\prasetia\\projects\\Animations\\Poster-2023-08-05\\angles.eps', format='eps')


        plt.show()

    def plotMultipleHist(self, success_ball, success_racket, fail_ball, fail_racket):
        np.random.seed(2025)
        success_idx = np.random.choice(len(success_ball), len(fail_ball))
        success_ball = success_ball[success_idx]

        success_idx = np.random.choice(len(success_racket), len(fail_racket))
        success_racket = success_racket[success_idx]

        types = (["ball"]  * len(success_ball)) + (["ball"]  * len(fail_ball)) + (["racket"]  * len(success_racket)) + (["racket"]  * len(fail_racket))
        labels = (["success"]  * len(success_ball)) + (["fail"]  * len(fail_ball)) + (["success"]  * len(success_racket)) + (["fail"]  * len(fail_racket))
        acc = np.hstack([success_ball, fail_ball, success_racket, fail_racket])

        df = pd.DataFrame({"acceleration": acc, "type": types, "label": labels})
        g = sns.catplot(
            data=df, x="type", y="acceleration", hue="label",
            kind="violin", split=True,
        )

        g.set(xlabel=None)
        g.set(ylabel=None)
        g._legend.remove()
        plt.savefig('F:\\users\\prasetia\\projects\\Animations\\Poster-2023-08-05\\ball_racket_acc.eps', format='eps')

        plt.show()

    def plotMultipleAxisHist(self, success_data, failure_data, x_axis="distance"):
        np.random.seed(2025)

        success_idx = np.random.choice(len(success_data), len(failure_data))
        success_data = success_data[success_idx]

        plt.figure(1)
        types = (["x", "y", "z"] * len(success_data)) + (["x", "y", "z"] * len(failure_data))
        labels = (["success"] * (3 * len(success_data))) + (["fail"] * (3 * len(failure_data)))
        dist = np.hstack([success_data.flatten(), failure_data.flatten()])

        df = pd.DataFrame({x_axis: dist, "axis": types, "label": labels})
        sns.catplot(
            data=df, x=x_axis, y="axis", hue="label",
            kind="violin", split=True
        )

        # dist = np.vstack([success_data, failure_data])
        # labels = (["success"] * (len(success_data))) + (["fail"] * (len(failure_data)))
        # df = pd.DataFrame({"x": dist[:, 0], "y": dist[:, 1], "z": dist[:, 2], "label": labels})
        # sns.pairplot(
        #     data=df,  hue="label",
        #
        # )


        # plt.savefig('F:\\users\\prasetia\\projects\\Animations\\Poster-2023-08-05\\distance_axis.eps', format='eps')
        plt.show()

    def plotLine(self, success_y, fail_y):
        np.random.seed(2025)


        success_idx = np.random.choice(len(success_y), len(fail_y))
        success_y = success_y[success_idx]

        n = success_y.shape[-1]
        n_2 = int(n/2)

        # time_point = (np.arange(-n, 0).astype(str).tolist() * len(success_y)) + (
        #             np.arange(-n, 0).astype(str).tolist() * len(fail_y))

        time_point = (np.arange(-n_2, n_2).astype(str).tolist() * len(success_y)) + (
                    np.arange(-n_2, n_2).astype(str).tolist() * len(fail_y))
        labels = (["success"] * (n * len(success_y))) + (["fail"] * (n * len(fail_y)))

        data = np.concatenate([success_y.flatten(), fail_y.flatten()])
        df = pd.DataFrame({"time_point": [float(i) for i in time_point], "acceleration_racket": data, "label": labels})
        plt.figure(1)
        g = sns.lineplot(data=df, x="time_point", y="acceleration_racket", hue="label", estimator="mean")
        sns.despine(offset=10, trim=True, left=True)
        # plt.vlines(0, 0, 10, linestyles="dotted", colors="r")
        # g.set_ylim([0, 5])
        g.set(xlabel=None)
        g.set(ylabel=None)

        g.legend().remove()
        # plt.savefig('F:\\users\\prasetia\\projects\\Animations\\Poster-2023-08-05\\accelerate_decelarate.eps', format='eps')
        plt.show()

    def plotFourLines(self, success_y, fail_y, prev_success_y, prev_fail_y):
        np.random.seed(2023)

        success_idx = np.random.choice(len(success_y), len(fail_y))
        success_y = success_y[success_idx]

        prev_success_idx =  np.random.choice(len(prev_success_y), len(prev_fail_y))
        prev_success_y = prev_success_y[prev_success_idx]


        n = success_y.shape[-1]
        n_2 = int(n / 2)

        # time_point = (np.arange(-n, 0).astype(str).tolist() * len(success_y)) + \
        #              (np.arange(-n, 0).astype(str).tolist() * len(fail_y)) +\
        #              (np.arange(-n, 0).astype(str).tolist() * len(prev_success_y)) + \
        #              (np.arange(-n, 0).astype(str).tolist() * len(prev_fail_y))

        time_point = (np.arange(-n_2, n_2).astype(str).tolist() * len(success_y)) + (
                    np.arange(-n_2, n_2).astype(str).tolist() * len(fail_y)) +  (
                    np.arange(-n_2, n_2).astype(str).tolist() * len(prev_success_y)) + (
                    np.arange(-n_2, n_2).astype(str).tolist() * len(prev_fail_y))
        labels = (["success"] * (n * len(success_y))) + (["fail"] * (n * len(fail_y))) + (["pre_success"] * (n * len(prev_success_y))) + \
                 (["prev_fail"] * (n * len(prev_fail_y)))

        data = np.concatenate([success_y.flatten(), fail_y.flatten(), prev_success_y.flatten(), prev_fail_y.flatten()])

        df = pd.DataFrame({"time_point": [float(i) for i in time_point], "acceleration_racket": data, "label": labels})
        plt.figure(1)
        g = sns.lineplot(data=df, x="time_point", y="acceleration_racket", hue="label", estimator="mean")
        sns.despine(offset=10, trim=True, left=True)

        g.set(xlabel=None)
        g.set(ylabel=None)


        # plt.savefig('F:\\users\\prasetia\\projects\\Animations\\Poster-2023-08-05\\accelerate_decelarate.eps', format='eps')
        plt.show()

    def plotLine2(self, success_y, fail_y, success_event, failure_event):
        np.random.seed(2023)


        ev =  -1 * int(np.average(np.concatenate([success_event, failure_event])))


        success_idx = np.random.choice(len(success_y), len(fail_y))
        success_y = success_y[success_idx]
        # fail_y = fail_y[:, ev:]

        n = success_y.shape[-1]


        time_point = (np.arange(-n, 0).astype(str).tolist() * len(success_y)) + (np.arange(-n, 0).astype(str).tolist() * len(fail_y))
        labels = (["success"] * (n * len(success_y))) + (["fail"] * (n * len(fail_y)))

        data = np.concatenate([success_y.flatten(), fail_y.flatten()])
        df = pd.DataFrame({"time_point": [float(i) for i in time_point], "acceleration_racket": data, "label": labels})
        plt.figure(2)
        g = sns.lineplot(data=df, x="time_point", y="acceleration_racket", hue="label")
        sns.despine(offset=10, trim=True, left=True)
        g.set(xlabel=None)
        g.set(ylabel=None)
        g.legend().remove()
        plt.vlines(int(ev), -1, 500, linestyles="dotted", colors="r")
        # plt.vlines(int(se), -1, 500, linestyles="dotted", colors="r")
        # plt.vlines(int(fe), -1, 500, linestyles="dotted", colors="g")
        # rect_se = mpatches.Rectangle((se, -1), se_w, 500,
        #                           # fill=False,
        #                           alpha=0.1,
        #                           facecolor="green")
        #
        # rect_fe = mpatches.Rectangle((fe, -1), fe_w, 500,
        #                           # fill=False,
        #                           alpha=0.1,
        #                           facecolor="red")
        # plt.gca().add_patch(rect_se)
        # plt.gca().add_patch(rect_fe)
        # plt.savefig('F:\\users\\prasetia\\projects\\Animations\\Poster-2023-08-05\\acc.eps', format='eps')
        plt.show()

    def plotMultiCategori(self, success, fail, prev_success, prev_fail):
        np.random.seed(2023)
        success_idx = np.random.choice(len(success), len(fail))
        success = success[success_idx]

        prev_success_idx = np.random.choice(len(prev_success), len(prev_fail))
        prev_success = prev_success[prev_success_idx]


        print(np.average(success))
        print(np.average(fail))
        print(np.average(prev_success))
        print(np.average(prev_fail))
        labels = (["success"] * len(success)) + (["fail"] *  len(fail)) + (["prev_success"] *  len(prev_success))  + (["prev_fail"] *  len(prev_fail))

        data = np.concatenate([success.flatten(), fail.flatten(), prev_success.flatten(), prev_fail.flatten()])

        df = pd.DataFrame({"RT": data, "label": labels})

        g = sns.boxplot(x='label', y='RT', data=df, whis=[0, 100], width=.6)
        # Add in points to show each observation
        sns.stripplot(x="label", y="RT", data=df,
                      size=4, color=".3", linewidth=0)
        # g = sns.displot(x='RT', hue='label', data=df, kind="kde", fill=True)


        sns.despine(offset=10, trim=True, left=True)
        g.set(xlabel=None)
        g.set(ylabel=None)


        plt.show()


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
        "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2023-02-08_A\\2023-02-08_A_T02_complete.pkl",
        "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2023-02-08_A\\2023-02-08_A_T04_complete.pkl",
        "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2023-02-08_A\\2023-02-08_A_T03_complete.pkl"
        ]

#     paths = [
#
#              "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-09_A\\2022-11-09_A_T04_complete.pkl",
# ]

    s_w_list = []
    s_t_list = []
    s_a_list = []
    s_d_list = []
    s_draw_list = []
    s_vb_list = []
    s_vr_list = []
    s_rall_list = []
    s_var_list = []
    s_do_list = []
    s_arm_list = []
    s_dracket_list = []
    s_gba_list = []
    s_gbar_list = []
    s_impct_list = []
    s_rt_list = []

    s_pc_impct_list = []
    s_p_rall_list = []
    s_p_dracket_list = []
    s_p_rt_list = []
    s_p_gba_list = []

    f_w_list = []
    f_t_list = []
    f_a_list = []
    f_d_list = []
    f_draw_list = []
    f_vb_list = []
    f_vr_list = []
    f_rall_list = []
    f_var_list = []
    f_do_list = []
    f_arm_list = []
    f_dracket_list = []
    f_gba_list = []
    f_gbar_list = []
    f_impct_list = []
    f_rt_list = []

    f_pc_impct_list = []
    f_p_rall_list = []
    f_p_dracket_list = []
    f_p_rt_list = []
    f_p_gba_list = []

    #events
    bts_list = []
    bws_list = []
    btf_list =  []
    bwf_list = []

    for p in paths:
        obj, sub, ball, tobii = reader.extractData(p)
        racket = None
        table = None
        for o in obj:
            if "racket" in o["name"].lower():
                racket = o
            if "table" in o["name"].lower():
                table = o
        print(p)
        feature = Classic(sub[0], racket, ball[0], tobii[0], table)
        bts, bws, btf, bwf = feature.extractEvents()
        s_wall, s_table, f_wall, f_table = feature.extractBouncePoints()
        s_a, f_a = feature.extractAnglePersonBall()
        s_d, f_d, s_draw, f_draw = feature.extractDistanceBounceEnd()
        s_vb, s_vr, f_vb, f_vr, s_rall, f_rall = feature.extractVelocityBallRacket()
        s_var, f_var =  feature.extractVarinceMovementBI()
        s_do, f_do = feature.extractDistanceBeforeImpact(n=5)
        s_arm, f_arm = feature.extractAnglesBeforeImpact(n=30)
        s_dracket, f_dracket, s_rt, f_rt = feature.extractDistanceRacketBeforeImpact(n=70)
        s_gb_a, f_gb_a = feature.extractGazeBallAngle(n=20, relative=False)
        s_gbr_a, f_gbr_a = feature.extractGazeBallAngle(n=20, relative=True)
        s_impct, f_impct = feature.extractImpactPositions()



        s_pc_impct, f_pc_impct = feature.extractPrevCurrentImpactPositions()
        s_p_vb, s_p_vr, f_p_vb, f_p_vr, s_p_rall, f_p_rall = feature.extractPrevVelocityBallRacket()
        s_p_dracket, f_p_dracket, s_p_rt, f_p_rt = feature.extractPrevDistanceRacketBeforeImpact(n=70)
        s_p_gba, f_p_gba = feature.extractPrevGazeBallAngle(n=20, relative=False)

        #add events
        bts_list.append(bts)
        bws_list.append(bws)
        btf_list.append(btf)
        bwf_list.append(bwf)

        s_t_list.append(s_table)
        s_w_list.append(s_wall)
        s_a_list.append(s_a)
        s_d_list.append(s_d)
        s_draw_list.append(s_draw)
        s_vb_list.append(s_vb)
        s_vr_list.append(s_vr)
        s_rall_list.append(s_rall)
        s_var_list.append(s_var)
        s_do_list.append(s_do)
        s_arm_list.append(s_arm)
        s_gba_list.append(s_gb_a)
        s_gbar_list.append(s_gbr_a)
        s_impct_list.append(s_impct)
        s_dracket_list.append(s_dracket)
        s_rt_list.append(s_rt)

        s_pc_impct_list.append(s_pc_impct)
        s_p_rall_list.append(s_p_rall)
        s_p_dracket_list.append(s_p_dracket)
        s_p_rt_list.append(s_p_rt)
        s_p_gba_list.append(s_p_gba)

        f_t_list.append(f_table)
        f_w_list.append(f_wall)
        f_a_list.append(f_a)
        f_d_list.append(f_d)
        f_draw_list.append(f_draw)
        f_vb_list.append(f_vb)
        f_vr_list.append(f_vr)
        f_rall_list.append(f_rall)
        f_var_list.append(f_var)
        f_do_list.append(f_do)
        f_arm_list.append(f_arm)
        f_gba_list.append(f_gb_a)
        f_gbar_list.append(f_gbr_a)
        f_impct_list.append(f_impct)
        f_dracket_list.append(f_dracket)
        f_rt_list.append(f_rt)

        f_pc_impct_list.append(f_pc_impct)
        f_p_rall_list.append(f_p_rall)
        f_p_dracket_list.append(f_p_dracket)
        f_p_rt_list.append(f_p_rt)
        f_p_gba_list.append(f_p_gba)


    wall_tr = None
    table_tr = None

    for o in obj:
        if o["name"].lower() == "table":
            table_tr = np.average(o["trajectories"].values, axis=0)
        if o["name"].lower() == "wall":
            wall_tr = np.average(o["trajectories"].values, axis=0)

    # visual.plot3D(np.concatenate(s_impct_list, 0), np.concatenate(f_impct_list, 0))

    # visual.plotWallHist(np.concatenate(s_w_list, 0), np.concatenate(f_w_list, 0), wall_tr)
    #
    # visual.plotWallHist(np.concatenate(s_t_list, 0), np.concatenate(f_t_list, 0), wall_tr)

    # visual.plotLine(np.concatenate(s_arm_list, 0), np.concatenate(f_arm_list, 0))

    # visual.plotLine(np.concatenate(s_dracket_list, 0), np.concatenate(f_dracket_list, 0))

    # visual.plotLine(np.concatenate(s_gba_list, 0), np.concatenate(f_gba_list, 0))
    visual.plotLine(np.concatenate(s_gbar_list, 0), np.concatenate(f_gbar_list, 0))

    # visual.plotLine(np.concatenate(s_rall_list, 0), np.concatenate(f_rall_list, 0))

    # visual.plotFourLines(np.concatenate(s_dracket_list, 0), np.concatenate(f_dracket_list, 0), np.concatenate(s_p_dracket_list, 0), np.concatenate(f_p_dracket_list, 0))

    # visual.plotMultiCategori(np.concatenate(s_rt_list, 0), np.concatenate(f_rt_list, 0), np.concatenate(s_p_rt_list, 0), np.concatenate(f_p_rt_list, 0))

    # visual.plotFourLines(np.concatenate(s_gba_list, 0), np.concatenate(f_gba_list, 0),
    #                      np.concatenate(s_p_gba_list, 0), np.concatenate(f_p_gba_list, 0))

    # visual.plotFourLines(np.concatenate(s_rall_list, 0), np.concatenate(f_rall_list, 0), np.concatenate(s_p_rall_list, 0), np.concatenate(f_p_rall_list, 0))

    #
    # visual.plotLine2(np.concatenate(s_do_list, 0), np.concatenate(f_do_list, 0), np.concatenate(bws_list, 0), np.concatenate(bwf_list, 0))

    # visual.plotTwoHist(np.concatenate(s_a_list, 0),
    #                      np.concatenate(f_a_list, 0))

    # visual.plotTwoHist(np.concatenate(s_d_list, 0),
    #                    np.concatenate(f_d_list, 0), y="distance")
    #
    # visual.plotTwoHist(np.concatenate(s_var_list, 0),
    #                    np.concatenate(f_var_list, 0), y="racket_variance")

    # visual.plotMultipleAxisHist(np.concatenate(s_impct_list, 0), np.concatenate(f_impct_list, 0), x_axis="impact_position")
    # visual.plotMultipleAxisHist(np.concatenate(s_pc_impct_list, 0), np.concatenate(f_pc_impct_list, 0), x_axis="distance_prev_current")
    # visual.plotMultipleAxisHist(np.concatenate(s_draw_list, 0), np.concatenate(f_draw_list, 0))
    #
    # visual.plotMultipleHist(np.concatenate(s_vb_list, 0),np.concatenate(s_vr_list, 0),
    #                      np.concatenate(f_vb_list, 0),   np.concatenate(f_vr_list, 0))
    # from scipy import stats
    #
    # print(stats.ttest_ind(np.concatenate(s_vr_list, 0), np.concatenate(f_vr_list, 0)))
    # visual.plotAngleHist(np.concatenate(s_v_list, 0),
    #                      np.concatenate(f_v_list, 0))
