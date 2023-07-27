import numpy as np
from Utils.Lib import wienerFilter, movingAverage, savgolFilter, cartesianToSpher
from FeaturesEngineering.GazeEvent import detectALPhase1, detectALPhase2, saccadeFeatures, detectSaccade, groupingFixation
from FeaturesEngineering.FeaturesLib import computeSegmentAngles, computeVelAcc
import pandas as pd
from Utils.DataReader import TobiiReader
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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
        table = np.nanmean(self.table_T, 0)
        wall = np.nanmean(self.wall_T, 0)

        # remove failure episodes from success
        if len(self.failures_idx) > 0:
            self.success_idx = self.success_idx[~np.in1d(self.success_idx[:, 1], self.failures_idx[:, 1])]
            # Get previous episodes
            self.prev_s_idx = np.argwhere(np.isin(self.success_idx[:, 1], self.success_idx[:, 0]))
            self.prev_f_idx = np.argwhere(np.isin(self.success_idx[:, 1], self.failures_idx[:, 0]))
            self.all_episodes = np.vstack([self.success_idx, self.failures_idx])
        else:
            self.all_episodes = self.success_idx
        # get segments
        self.wall_segment = np.nanmean(wall, axis=0, keepdims=True)
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
        self.ball_t = self.smoothBall(ball["trajectories"].values)

        # normalize tobii dta
        self.normalizeTobii()

    def normalizeTobii(self):
        for e in self.all_episodes :
            start = e[0] - 10
            stop = e[1] + 10

            self.left_gaze_dir[start:stop] = self.tobii_reader.gapFill(self.left_gaze_dir[start:stop])
            self.right_gaze_dir[start:stop] = self.tobii_reader.gapFill(self.right_gaze_dir[start:stop])

            self.gaze_point[start:stop] = self.tobii_reader.gapFill(self.gaze_point[start:stop])

    def smoothBall(self, ball: np.array) -> np.array:
        ball = np.array([movingAverage(ball[:, i], n=1) for i in range(3)]).transpose()
        return ball

    def eventToLabels(self, e_idx: int = 0):
        if e_idx == 0:
            return "Ball_Hit"
        elif e_idx == 2:
            return "Bounce_Wall"
        elif e_idx == 3:
            return "Bounce_Table"
        elif e_idx == 1:
            return "Ball_Impact"
        else:
            return "Unknown"

    def distanceBallBeforeEvent(self, ball: np.array, segment: np.array, episodes: np.array, e_idx: int = 1,
                                n: int = 10, ball_last=False):
        '''
        how a segment changes over time in respect to the last ball
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
        segment_bfe = segment[episodes[:, e_idx] - n]  # position of the segment n frame before the event occurs

        if ball_last:
            ball_bfe = ball[episodes[:, e_idx]]  # position of the ball when the segment occurs

        else:
            ball_bfe = ball[episodes[:, e_idx] - n]  # position of the segment n frame before the event occurs

        d_bfe = np.linalg.norm(ball_bfe - segment_bfe, axis=-1)
        raw_axis_dist = np.sqrt(np.square(segment_bfe - ball_bfe))

        features_summary = {"event": self.eventToLabels(e_idx),
                            "d_before_event": d_bfe,
                            "d_axis": raw_axis_dist
                            }

        return features_summary

    def distanceBallOverTime(self, ball: np.array, segment, episode, e_idx: int = 1, n: int = 15, ball_relative=False):
        '''
        :param ball: ball trajectory vector
        :param episodes: episodes
        :param segment: segment trajectory vector
        :param e_idx: index of event
        :param n: n frames before event
        :return:
         - event name
         - distance in overtime before the event
        '''
        dists = []
        for e in episode:
            start = e[e_idx] - n
            stop = e[e_idx]+1

            segment_t = segment[start:stop]
            if ball_relative:
                ball_t = ball[stop]
            else:
                ball_t = ball[start:stop]
            dist = np.linalg.norm(ball_t - segment_t, axis=-1)
            dists.append(dist)

        features_summary = {
            "event": self.eventToLabels(e_idx),
            "d_ot_event": np.vstack(dists).astype(float),
        }
        return features_summary

    def saccadePursuit(self, episode, th_al_p1=10, th_al_p2=25, th_angle_p=15, normalize=True):
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

        # saccades features
        saccades_phase1_list = np.zeros((len(episode), 9, 7))
        saccades_phase2_list = np.zeros((len(episode), 9, 7))
        saccades_phase3_list = np.zeros((len(episode), 9, 7))

        ball_n = self.ball_t

        gaze_n = self.tobii_reader.local2GlobalGaze(self.gaze_point, self.tobii_segment_T, self.tobii_segment_R,
                                                  translation=True)



        p1_features_list = []
        p2_features_list = []
        p3_features_list = []
        i=0
        for se in episode :
            # phase 1
            p1_s = se[0]
            p1_e = se[2]

            # phase 2
            p2_s = se[2]
            p2_e = se[3]

            # phase 3
            p3_s = se[3]
            p3_e = se[1]


            # detect saccade
            gaze = np.array([movingAverage(gaze_n[p1_s:p3_e, i], n=1) for i in range(3)]).transpose()
            ball = np.array([movingAverage(ball_n[p1_s:p3_e, i], n=1) for i in range(3)]).transpose()
            tobii = np.array([movingAverage(self.tobii_segment_T[p1_s:p3_e, i], n=1) for i in range(3)]).transpose()
            tobii_avg = np.array([movingAverage(self.tobii_segment_T[p1_s:p3_e, i], n=2) for i in range(3)]).transpose()
            onset_offset_saccade, onset_offset_sp, onset_offset_fix, stream_label = detectSaccade(gaze, ball, tobii, tobii_avg)

            import matplotlib.pyplot as plt

            _, gaze_az, gaze_elv = cartesianToSpher(vector=gaze - tobii_avg, swap=False)
            _, ball_az, ball_elv = cartesianToSpher(vector=ball, swap=False)
            gaze_plot = np.vstack([gaze_az, gaze_elv]).transpose()
            ball_plot = np.vstack([ball_az, ball_elv]).transpose()
            x = gaze_plot[:, 0]
            y = gaze_plot[:, 1]
            x_ball = ball_plot[:, 0]
            y_ball = ball_plot[:, 1]


            print(self.tobii_time[p1_s])


            plt.plot(x, y, "-o")

            plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=2, width = 0.002)

            # plt.plot(x_ball, y_ball, "-*")
            # plt.quiver(x_ball[:-1], y_ball[:-1], x_ball[1:] - x_ball[:-1], y_ball[1:] - y_ball[:-1], scale_units='xy',
            #            angles='xy', scale=2,
            #            width=0.002)


            for on, off in onset_offset_sp:
                plt.scatter(x[on:off +1], y[on:off + 1],  color="blue", zorder=2)

            for on, off in onset_offset_fix:
                plt.scatter(x[on:off + 1], y[on:off + 1],  color="black", zorder=3)

            for on, off in onset_offset_saccade:
                plt.scatter(x[on:off + 1], y[on:off + 1], color="red", zorder=4)


            plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            color_labels = ["black", "blue", "red"]
            ax.scatter(ball[:, 0], ball[:, 1], ball[:, 2], alpha=0.01, depthshade=False)
            # ax.scatter(gaze[:, 0], gaze[:, 1], gaze[:, 2])

            for i in range(3):
                idx = np.argwhere(stream_label == (i+1))
                ax.scatter(ball[idx, 0], ball[idx, 1], ball[idx, 2], color=color_labels[i], zorder=4,
                           alpha=1, depthshade=False)





            plt.show()

            # saccades_phase1 = saccadeFeatures(self.eye_event[p1_s:p1_e], gaze_an[p1_s:p1_e], ball_an[p1_s:p1_e])
            # saccades_phase2 = saccadeFeatures(self.eye_event[p2_s:p2_e+3], gaze_an[p2_s:p2_e+3], ball_an[p2_s:p2_e+3])
            # saccades_phase3 = saccadeFeatures(self.eye_event[p3_s:p3_e], gaze_an[p3_s:p3_e], ball_an[p3_s:p3_e])
            #
            # if len(saccades_phase1) > 0:
            #     try:
            #         saccades_phase1_list[i, :len(saccades_phase1)] = saccades_phase1
            #     except:
            #         print("eeror")
            # if len(saccades_phase2) > 0:
            #     saccades_phase2_list[i, :len(saccades_phase2)] = saccades_phase2
            # if len(saccades_phase3) > 0:
            #     saccades_phase3_list[i, :len(saccades_phase3)] = saccades_phase3
            # saccades_phase1_list.append(saccades_phase1)
            # saccades_phase2_list.append(saccades_phase2)
            # saccades_phase3_list.append(saccades_phase3)

        #     phase1_features = detectALPhase1(self.eye_event[p1_s:p1_e], az[p1_s:p1_e], elv[p1_s:p1_e], th=th_al_p1)
        #     phase2_features, phase3_features = detectALPhase2(self.eye_event[p2_s:p2_e], self.eye_event[p3_s:p3_e],
        #                                                           gaze_an[p2_s:p3_e], ball_an[p2_s:p3_e], th=th_al_p2,
        #                                                           th_angle_p=th_angle_p)
        #
        #
        #     p1_features_list.append(phase1_features)
        #     p2_features_list.append(phase2_features)
        #     p3_features_list.append(phase3_features)
        #     i+=1
        #
        # p1_features_list = np.vstack(p1_features_list)
        # p2_features_list = np.vstack(p2_features_list)
        # p3_features_list = np.vstack(p3_features_list)
        #
        # if normalize:
        #     features_summary = {
        #         "al_p1_percentage": np.average(p1_features_list[:, 0] != 1e+4),
        #         "al_p2_percentage": np.average(p2_features_list[:, 0] != 1e+4),
        #         "pr_p2_percentage": np.average(p3_features_list[:, 0] != 1e+4),
        #         "p1_features": p1_features_list[p1_features_list[:, 0] != 1e+4],
        #         "p2_features": p2_features_list[p2_features_list[:, 0] != 1e+4],
        #         "p3_features": p3_features_list[p3_features_list[:, 0] != 1e+4],
        #         "saccade_p1": saccades_phase1_list,
        #         "saccade_p2": saccades_phase2_list,
        #         "saccade_p3": saccades_phase3_list,
        #
        #     }
        # else:
        #     features_summary = {
        #         "al_p1_percentage": np.average(p1_features_list[:, 0] != 1e+4),
        #         "al_p2_percentage": np.average(p2_features_list[:, 0] != 1e+4),
        #         "pr_p2_percentage": np.average(p3_features_list[:, 0] != 1e+4),
        #         "p1_features": p1_features_list,
        #         "p2_features": p2_features_list,
        #         "p3_features": p3_features_list,
        #         "saccade_p1": saccades_phase1_list,
        #         "saccade_p2": saccades_phase2_list,
        #         "saccade_p3": saccades_phase3_list,
        #     }
        #
        # return features_summary



    def gazeBallAngleBeforeEvent(self, episodes=None, e_idx=1, n: int = 15):
        '''
        :param episodes: episodes
        :param n: n frame before the event
        :param e_idx: event index
        :return:
        - event name
        - gaze-ball angle
        '''

        ball_n = self.tobii_reader.global2LocalGaze(self.ball_t, self.tobii_segment_T, self.tobii_segment_R,
                                                    translation=True)

        gaze_n = self.gaze_point

        _, b_az, b_elv = cartesianToSpher(ball_n, swap=True)
        _, g_az, g_elv = cartesianToSpher(gaze_n, swap=True)

        az = g_az - b_az
        elv = g_elv - b_elv

        dist_angle = np.sqrt(np.square(az) + np.square(elv))
        angle_list = []
        for e in episodes:
            start = e[e_idx] - n
            stop = e[e_idx] + n
            angles_t = dist_angle[start: stop]
            angle_list.append(angles_t)

        features_summary = {
            "event": self.eventToLabels(e_idx),
            "g_ba": np.asarray(angle_list, dtype=float)
        }

        return features_summary

    def angleAtEvent(self, seg1: np.array, seg2: np.array, episodes: np.array, e_idx: int = 2):
        '''
        :param seg1: trajectory of segment 1
        :param seg2: trajectory of segment 2
        :param episodes: episodes
        :param e_idx: index of the event
        :return:
        - event name
        - angle at that moment
        '''

        seg1_event = seg1[episodes[:, e_idx]]
        seg2_event = seg2[episodes[:, e_idx]]

        angles = computeSegmentAngles(seg1_event, seg2_event)

        features_summary = {
            "event": self.eventToLabels(e_idx),
            "angles": np.array(angles)
        }

        return features_summary

    def segmentPositionAtEvent(self, seg: np.array, episode: np.array, e_idx: int = 1, label="position"):
        position_list = []
        for e in episode:
            position_list.append(seg[e[e_idx]])

        features_summary = {
            "event": self.eventToLabels(e_idx),
            label: np.asarray(position_list)
        }

        return features_summary

    def prevCurrectAtEvent(self, v: np.array, ref_episodes: np.array, current_episodes: np.array,
                           prev_episodes: np.array, e_idx, label="position"):
        prev_list = []
        current_list = []
        for i in prev_episodes:
            prev_e = ref_episodes[i][0]
            curr_e = current_episodes[current_episodes[:, 0] == prev_e[1]][0]
            prev_impact = v[prev_e[e_idx]]
            curr_impact = v[curr_e[e_idx]]

            prev_list.append(prev_impact)
            current_list.append(curr_impact)
        features_summary = {
            "event": self.eventToLabels(e_idx),
            "prev_"+label: np.asarray(prev_list),
            "current_"+label: np.asarray(current_list)
        }
        return features_summary

    def velocityAcceleration(self, seg:np.array, episode:np.array, e_idx:int=1, n:int=15, avg=False):
        speed_list = []
        vel_list = []
        acc_list = []
        for e in episode:
            start = e[e_idx] - (n + 2)
            stop = e[e_idx] + 1
            seg_part = seg[start:stop]
            speed, vel, acc = computeVelAcc(seg_part)

            if avg:
                speed_list.append(np.average(speed))
                vel_list.append(np.average(vel))
                acc_list.append(np.average(acc))
            else:
                speed_list.append(speed)
                vel_list.append(vel)
                acc_list.append(acc)

        features_summary = {
            "event": self.eventToLabels(e_idx),
            "speed": np.asarray(speed_list),
            "velocity": np.asarray(vel_list),
            "acceleration": np.asarray(acc_list)
        }

        return features_summary


    def startForwardSwing(self, episode:np.array):
        dist_wall = np.linalg.norm(self.table_segment[:, 1:] - self.racket_segment_T[:, 1:], axis=-1)
        rt_list = []
        for e in episode:
            start = e[0]
            end = e[1]

            rt_dist = dist_wall[start:end]
            peaks, _ = find_peaks(rt_dist, distance=50)

            if len(peaks) == 0:
                print(e)
                racket_split = np.diff(self.racket_segment_T[start:end, 1], n=1) *  -1
                racket_split = np.append(racket_split[0], racket_split)
                peaks, _ = find_peaks(racket_split, distance=50)
                if len(peaks) > 0:
                    rt = (e[1] - e[0]) - peaks[-1]
                    rt_list.append(rt)
                else:
                    rt_list.append(1e+4)
                # plt.subplot(2, 1, 1)
                # plt.plot(racket_split)
                # plt.vlines(peaks, np.min(racket_split), np.max(racket_split), colors="red")
                # plt.subplot(2, 1, 2)
                # plt.plot(dist_wall[start:end])
                # plt.vlines(peaks, np.min(dist_wall[start:end]), np.max(dist_wall[start:end]), colors="red")
                # plt.show()

            else:
                rt = (e[1] - e[0]) - peaks[-1]
                rt_list.append(rt)

        features_summary = {
            "start_fs": np.asarray(rt_list),
        }

        return features_summary
    def extractBallContactPosition(self, prev=False):
        if prev:
            success_positions = self.prevCurrectAtEvent(self.ball_t, self.success_idx, self.success_idx, self.prev_s_idx, 1)
            failure_position = self.prevCurrectAtEvent(self.ball_t, self.success_idx, self.failures_idx, self.prev_f_idx, 1)

        else:
            success_positions = self.segmentPositionAtEvent(self.ball_t, self.success_idx, 1)
            failure_position = self.segmentPositionAtEvent(self.ball_t, self.failures_idx, 1)

        return success_positions, failure_position


    def extractRacketVelocity(self, e_idx: int=1, n:int=15, prev=False):

        if prev:
            success_s_v_a = self.velocityAcceleration(self.racket_segment_T, self.success_idx[self.prev_s_idx[:, 0]], e_idx, n=n)
            failures_s_v_a = self.velocityAcceleration(self.racket_segment_T, self.success_idx[self.prev_f_idx[:, 0]], e_idx, n=n)
        else:
            success_s_v_a = self.velocityAcceleration(self.racket_segment_T, self.success_idx, e_idx, n=n)
            failures_s_v_a = self.velocityAcceleration(self.racket_segment_T, self.failures_idx, e_idx, n=n)

        return success_s_v_a, failures_s_v_a

    def extractRWirst(self, e_idx: int=1, n:int=15, prev=False):

        if prev:
            success_s_v_a = self.velocityAcceleration(self.racket_segment_R, self.success_idx[self.prev_s_idx[:, 0]], e_idx, n=n, avg=True)
            failures_s_v_a = self.velocityAcceleration(self.racket_segment_R, self.success_idx[self.prev_f_idx[:, 0]], e_idx, n=n, avg=True)
        else:
            success_s_v_a = self.velocityAcceleration(self.racket_segment_R, self.success_idx, e_idx, n=n, avg=True)
            failures_s_v_a = self.velocityAcceleration(self.racket_segment_R, self.failures_idx, e_idx, n=n, avg=True)

        return success_s_v_a, failures_s_v_a


    def extractRacketBallAngleImpact(self, prev=False):
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(xs, ys, zs, marker=m)

        origin_racket = np.zeros_like(self.racket_segment_T)
        origin_racket[:, 2] = 1
        racket_ball_angle = computeSegmentAngles(self.ball_t - self.racket_segment_T, origin_racket)

        if prev:
            success_angles = self.prevCurrectAtEvent(racket_ball_angle, self.success_idx, self.success_idx,
                                                        self.prev_s_idx, 1, label="angle")
            failure_angles = self.prevCurrectAtEvent(racket_ball_angle, self.success_idx, self.failures_idx,
                                                       self.prev_f_idx, 1, label="angle")
        else:
            success_angles = self.segmentPositionAtEvent(racket_ball_angle, self.success_idx, 1, label="angle")
            failure_angles = self.segmentPositionAtEvent(racket_ball_angle, self.failures_idx, 1, label="angle")

        return success_angles, failure_angles


    def extractForwardswing(self, prev=False):

        if prev:
            success_rt = self.startForwardSwing(self.success_idx[self.prev_s_idx[:, 0]])
            failures_rt = self.startForwardSwing(self.success_idx[self.prev_f_idx[:, 0]])
        else:
            success_rt = self.startForwardSwing(self.success_idx)
            failures_rt = self.startForwardSwing(self.failures_idx)


        return success_rt, failures_rt



    def extractSaccadePursuit(self, normalize=True):
        success_rt = self.saccadePursuit(self.success_idx, normalize=normalize, th_angle_p=25)
        failures_rt = []
        if len(self.failures_idx)> 0:
            failures_rt = self.saccadePursuit(self.failures_idx, normalize=normalize, th_angle_p=25)

        return success_rt, failures_rt



    def extractGazeBallAngle(self, e_idx=1, n=20):

        success_angle = self.gazeBallAngleBeforeEvent(self.success_idx, e_idx=e_idx, n=n)
        failure_angle = self.gazeBallAngleBeforeEvent(self.failures_idx, e_idx=e_idx, n=n)

        return success_angle, failure_angle

