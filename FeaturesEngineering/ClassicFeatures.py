import numpy as np
from Utils.Lib import wienerFilter, movingAverage, savgolFilter, cartesianToSpher
from FeaturesEngineering.GazeEvent import saccadeFeatures, detectSaccade, fixationSPFeatures
from FeaturesEngineering.FeaturesLib import computeSegmentAngles, computeVelAcc
import pandas as pd
from Utils.DataReader import TobiiReader
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from FeaturesEngineering.Plotting import gazeEventsPlotting

class Classic:

    def __init__(self, sub:dict, racket:dict, ball:dict, tobii:dict, table:dict, wall:dict):
        '''
        :param sub: dictionary of subjects
        :param racket:dictionary of racket
        :param ball:dictionary of ball
        :param tobii:dictionary of tobii
        :param table:dictionary of table
        :param wall:dictionary of wall
        '''
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
        '''
        perform gap filling in Tobii
        :return:
        '''
        for e in self.all_episodes :
            start = e[0] - 10
            stop = e[1] + 10

            self.left_gaze_dir[start:stop] = self.tobii_reader.gapFill(self.left_gaze_dir[start:stop])
            self.right_gaze_dir[start:stop] = self.tobii_reader.gapFill(self.right_gaze_dir[start:stop])

            self.gaze_point[start:stop] = self.tobii_reader.gapFill(self.gaze_point[start:stop])

    def smoothBall(self, ball: np.array) -> np.array:
        '''
        :param ball: ball in the world
        :return: smoothed ball trajectory
        '''
        ball = np.array([movingAverage(ball[:, i], n=1) for i in range(3)]).transpose()
        return ball

    def eventToLabels(self, e_idx: int = 0) -> str:
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
                                n: int = 10, ball_last=False)-> dict:
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

    def distanceBallOverTime(self, ball: np.array, segment, episode, e_idx: int = 1, n: int = 15, ball_relative=False)-> dict:
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

    def detectGazeEvents(self, episode, th_al_p1=10, th_al_p2=25, th_angle_p=15, normalize=True)-> dict:
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
        anticipation_phase1_list = np.zeros(shape=(len(episode), 10))
        anticipation_phase2_list = np.zeros(shape=(len(episode), 10))
        anticipation_phase3_list = np.zeros(shape=(len(episode), 10))

        # attended fixation and sp features
        fsp_phase3_list = np.zeros(shape=(len(episode), 5))

        ball_n = self.ball_t

        gaze_n = self.tobii_reader.local2GlobalGaze(self.gaze_point, self.tobii_segment_T, self.tobii_segment_R,
                                                  translation=True)

        i=0

        for se in episode :
            # episode index (0:hit1, 1:hit2, 2:bounce_wall, 3:bounce_table)
            # p1 = hit1 -> bounce wall
            # p2 = bounce_wall -> bounce table
            # p3 = bounce_table -> hit2

            # phase 1
            p1_s = se[0]-3
            p1_e = se[2]

            # phase 2
            p2_s = se[2]
            p2_e = se[3]

            # phase 3
            p3_s = se[3]
            p3_e = se[1]


            # detect saccade
            gaze = np.array([movingAverage(gaze_n[p1_s:p3_e, i], n=2) for i in range(3)]).transpose()
            ball = np.array([movingAverage(ball_n[p1_s:p3_e, i], n=1) for i in range(3)]).transpose()
            tobii = np.array([movingAverage(self.tobii_segment_T[p1_s:p3_e, i], n=1) for i in range(3)]).transpose()
            tobii_avg = np.array([movingAverage(self.tobii_segment_T[p1_s:p3_e, i], n=3) for i in range(3)]).transpose()
            onset_offset_saccade, onset_offset_sp, onset_offset_fix, stream_label = detectSaccade(gaze, ball, tobii, tobii_avg)

            # normalize gaze and ball
            gaze_ih = gaze - tobii_avg
            ball_ih = ball - tobii_avg
            win_length = 10
            # do normalize the start and the end of the episode
            p1s, p1e = p1_s - p1_s, p1_e - p1_s
            p2s, p2e = p2_s - p1_s, p2_e - p1_s
            p3s, p3e = p3_s - p1_s, p3_e - p1_s

            if len(onset_offset_saccade) > 0:
                saccade_p1 = onset_offset_saccade[onset_offset_saccade[:, 0] <=p1e]
                saccade_p2 = onset_offset_saccade[(onset_offset_saccade[:, 0] > p2s)&(onset_offset_saccade[:, 0] <= p2e)]
                saccade_p3 = onset_offset_saccade[(onset_offset_saccade[:, 0] > p3s)]

                # print(p1_s)
                # print(self.tobii_time[p1_s - 1])
                # print(self.tobii_time[p1_s])
                # print(self.tobii_time[p1_s + 1])
                #
                # print(onset_offset_saccade)

                al_cs_labels = []
                # print("------------------------Phase1------------------------")
                if len(saccade_p1) > 0:
                    saccades_phase1, al_cs_p1 = saccadeFeatures(saccade_p1, gaze_ih, ball_ih, win_length=10, phase_start=p1s, phase_end=p1e)
                    al_cs_labels.append(al_cs_p1)
                    if np.sum(al_cs_p1 == 1) == 1:
                        anticipation_phase1_list[i] = saccades_phase1[al_cs_p1==1]

                # print("------------------------Phase2------------------------")
                if len(saccade_p2) > 0:
                    saccades_phase2, al_cs_p2 = saccadeFeatures(saccade_p2, gaze_ih, ball_ih,
                                                      win_length=10, phase_start=p2s, phase_end=p2e, phase_2=True)
                    al_cs_labels.append(al_cs_p2)
                    if np.sum(al_cs_p2 == 1) == 1:
                        anticipation_phase2_list[i] = saccades_phase2[al_cs_p2 == 1]
                        if len(onset_offset_sp) > 0:
                            fsp_phase3 = fixationSPFeatures(onset_offset_sp, saccade_p2[al_cs_p2 == 1], gaze_ih, ball_ih, phase_start=p3s, phase_end=p3e)
                            fsp_phase3_list[i] = fsp_phase3

                # print("------------------------Phase3------------------------")
                if len(saccade_p3) > 0:
                    if saccade_p3[-1, 1] > len(gaze):
                        saccade_p3[-1, 1] = len(gaze) - 1
                    win_length = len(ball) - saccade_p3[-1, 1]
                    saccades_phase3, al_cs_p3 = saccadeFeatures(saccade_p3, gaze_ih, ball_ih,
                                                      win_length=win_length, phase_start=p3s, phase_end=p3e, classify_al=False)

                    if np.sum(al_cs_p3 == 1) == 1:
                        anticipation_phase3_list[i] = saccades_phase3[al_cs_p3 == 1]
                    al_cs_labels.append(al_cs_p3)


                # if you want to plot the results
                # al_cs_labels = np.concatenate(al_cs_labels)
                # gazeEventsPlotting(gaze, tobii_avg, ball, onset_offset_saccade, onset_offset_sp, onset_offset_fix,
                #                    stream_label, al_cs_labels)

            i+=1



        features_summary = {
                "saccade_p1": anticipation_phase1_list,
                "saccade_p2": anticipation_phase2_list,
                "saccade_p3": anticipation_phase3_list,
                "fsp_phase3": fsp_phase3_list,
        }

        return features_summary



    def gazeBallAngleBeforeEvent(self, episodes=None, e_idx=1, n: int = 15)-> dict:
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

    def angleAtEvent(self, seg1: np.array, seg2: np.array, episodes: np.array, e_idx: int = 2)-> dict:
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

    def segmentPositionAtEvent(self, seg: np.array, episode: np.array, e_idx: int = 1, label="position")-> dict:
        '''
        :param seg: segment array
        :param episode: episode
        :param e_idx: index of the event in an episode (0: hit1, 1: hit2, 2: bounce on the wall, 3: bounce on the table)
        :param label: label of the features
        :return:
        '''
        position_list = []
        for e in episode:
            position_list.append(seg[e[e_idx]])

        features_summary = {
            "event": self.eventToLabels(e_idx),
            label: np.asarray(position_list)
        }

        return features_summary

    def prevCurrectAtEvent(self, v: np.array, ref_episodes: np.array, current_episodes: np.array,
                           prev_episodes: np.array, e_idx, label="position")-> dict:
        '''
        :param v: segment of trajectories
        :param ref_episodes: reference episodes
        :param current_episodes: current episodes
        :param prev_episodes: previous episodes
        :param e_idx: index of the event
        :param label: label of the event
        :return:
        '''
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

    def velocityAcceleration(self, seg:np.array, episode:np.array, e_idx:int=1, n:int=15, avg=False)-> dict:
        '''
        :param seg: segments
        :param episode: episodes
        :param e_idx: index of the event
        :param n: start of the segment whose velocity will be computed about
        :param avg: whether to perform average or not
        :return:
        '''
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


    def startForwardSwing(self, episode:np.array)-> dict:
        '''
        compute the start time of forward swing
        start of forward swing: total number of frames taken for individuals to hit the ball after it bounces on the table
        :param episode: episodes
        :return: the list of start time of forward swing
        '''
        vel_racket = np.linalg.norm(self.racket_segment_T[1:, ] - self.racket_segment_T[:-1, ], axis=-1)
        acc_racket = np.diff(np.pad(vel_racket, (1, 1), 'symmetric'), n=1, axis=0)
        rt_list = []
        for e in episode:
            start = e[2]
            end = e[1]

            rt_vel = acc_racket[start:end]
            peaks, _ = find_peaks(rt_vel, distance=10)
            # print()
            #
            # plt.subplot(2, 1, 1)
            # plt.plot(vel_racket[start:end])
            # plt.vlines(peaks, np.min(racket_split), np.max(racket_split), colors="red")
            # plt.subplot(2, 1, 2)
            # plt.plot(dist_wall[start:end])
            # plt.vlines(peaks, np.min(dist_wall[start:end]), np.max(dist_wall[start:end]), colors="red")
            # plt.show()


            if len(peaks) == 0:
                # print(e)
                # racket_split = np.diff(self.racket_segment_T[start:end, 1], n=1) *  -1
                # racket_split = np.append(racket_split[0], racket_split)
                # peaks, _ = find_peaks(racket_split, distance=50)
                # if len(peaks) > 0:
                #     rt = (e[1] - e[0]) - peaks[-1]
                #     rt_list.append(rt)
                # else:
                rt_list.append(np.nan)

            else:
                high_peaks = peaks[-1]
                # rt = (e[1] - e[2]) - high_peaks
                # rt = (e[1] - e[2]) - high_peaks
                rt_list.append(high_peaks)

        features_summary = {
            "avg_start_fs": np.nanmean(np.asarray(rt_list)),
            "std_start_fs": np.nanstd(rt_list),
        }

        return features_summary
    def extractBallContactPosition(self, prev=False)-> tuple:
        '''
        extract the position of the ball
        :param prev: whether consider previous episode or not
        :return: the position of the ball when individual succeed or fail
        '''
        if prev:
            success_positions = self.prevCurrectAtEvent(self.ball_t, self.success_idx, self.success_idx, self.prev_s_idx, 1)
            failure_position = self.prevCurrectAtEvent(self.ball_t, self.success_idx, self.failures_idx, self.prev_f_idx, 1)

        else:
            success_positions = self.segmentPositionAtEvent(self.ball_t, self.success_idx, 1)
            failure_position = self.segmentPositionAtEvent(self.ball_t, self.failures_idx, 1)

        return success_positions, failure_position


    def extractRacketVelocity(self, e_idx: int=1, n:int=15, prev=False)->tuple:
        '''
        :param e_idx: index of the event
        :param n: the start of the array before the event
        :param prev: whether consider previous episode or not
        :return: velocity and acceleration of the racket
        '''

        if prev:
            success_s_v_a = self.velocityAcceleration(self.racket_segment_T, self.success_idx[self.prev_s_idx[:, 0]], e_idx, n=n)
            failures_s_v_a = self.velocityAcceleration(self.racket_segment_T, self.success_idx[self.prev_f_idx[:, 0]], e_idx, n=n)
        else:
            success_s_v_a = self.velocityAcceleration(self.racket_segment_T, self.success_idx, e_idx, n=n)
            failures_s_v_a = self.velocityAcceleration(self.racket_segment_T, self.failures_idx, e_idx, n=n)

        return success_s_v_a, failures_s_v_a

    def extractRWirst(self, e_idx: int=1, n:int=15, prev=False)-> tuple:
        '''
        :param e_idx: index of the event
        :param n: the start of the array before the event
        :param prev: whether consider previous episode or not
        :return: velocity and accrelation of right wrist
        '''

        if prev:
            success_s_v_a = self.velocityAcceleration(self.racket_segment_R, self.success_idx[self.prev_s_idx[:, 0]], e_idx, n=n, avg=True)
            failures_s_v_a = self.velocityAcceleration(self.racket_segment_R, self.success_idx[self.prev_f_idx[:, 0]], e_idx, n=n, avg=True)
        else:
            success_s_v_a = self.velocityAcceleration(self.racket_segment_R, self.success_idx, e_idx, n=n, avg=True)
            failures_s_v_a = self.velocityAcceleration(self.racket_segment_R, self.failures_idx, e_idx, n=n, avg=True)

        return success_s_v_a, failures_s_v_a


    def extractRacketBallAngleImpact(self, prev=False) -> tuple:
        '''
        :param prev: whether consider previous episode or not
        :return: the angle between the racket and the ball at the impact
        '''
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
        '''
        :param prev: whether consider previous episode or not
        :return: the time when individuals start the forward swing
        '''

        if prev:
            success_rt = self.startForwardSwing(self.success_idx[self.prev_s_idx[:, 0]])
            failures_rt = self.startForwardSwing(self.success_idx[self.prev_f_idx[:, 0]])
        else:
            success_rt = self.startForwardSwing(self.success_idx)
            failures_rt = self.startForwardSwing(self.failures_idx)


        return success_rt, failures_rt



    def extractSaccadePursuit(self, normalize=True):
        '''
        :param normalize: whether to normalize the results or not
        :return: the features of anticipatory look
        '''
        success_rt = self.detectGazeEvents(self.success_idx, normalize=normalize, th_angle_p=25)
        failures_rt = []
        if len(self.failures_idx)> 0:
            failures_rt = self.detectGazeEvents(self.failures_idx, normalize=normalize, th_angle_p=25)

        return success_rt, failures_rt



    def extractGazeBallAngle(self, e_idx=1, n=20):
        '''
        :param e_idx: event index
        :param n: the n frame before the event
        :return: the angle between the gaze and the ball
        '''

        success_angle = self.gazeBallAngleBeforeEvent(self.success_idx, e_idx=e_idx, n=n)
        failure_angle = self.gazeBallAngleBeforeEvent(self.failures_idx, e_idx=e_idx, n=n)

        return success_angle, failure_angle

