import numpy as np
from Utils.Lib import wienerFilter, movingAverage, savgolFilter, cartesianToSpher
from FeaturesEngineering.GazeEvent import saccadeFeatures, detectSaccade, fixationSPFeatures, jointAttentionFeatures, gazeMovemenSynctFeatures
from Utils.DataReader import TobiiReader
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from Utils.Interpolate import interpolate
from FeaturesEngineering.Plotting import gazeEventsPlotting
from FeaturesEngineering.FeaturesLib import computeVectorsDirection, whichShoulder, computeSegmentAngles, computeVelAcc
from Analysis.Conf import RACKET_MASS, BALL_MASS
from SubjectObject import Subject, Ball, TableWall


class FeaturesExtractor:

    def __init__(self, sub: Subject, ball: Ball, table_wall: TableWall):
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

        self.p = sub
        self.b = ball
        self.tw = table_wall

    def detectGazeEvents(self, episode, th_al_p1=10, th_al_p2=25, th_angle_p=15, normalize=True) -> dict:
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
        anticipation_phase1_list = np.empty(shape=(len(episode), 13))
        anticipation_phase1_list[:] = np.nan
        anticipation_phase2_list = np.empty(shape=(len(episode), 13))
        anticipation_phase2_list[:] = np.nan
        anticipation_phase3_list = np.empty(shape=(len(episode), 13))
        anticipation_phase3_list[:] = np.nan
        gaze_event_list = np.zeros(shape=(len(episode), 5))

        # attended fixation and sp features
        fsp_phase3_list = np.empty(shape=(len(episode), 9))
        fsp_phase3_list[:] = np.nan

        ball_n = self.b.ball_t

        gaze_n = self.tobii_reader.local2GlobalGaze(self.p.gaze_point, self.p.tobii_segment_T, self.p.tobii_segment_R,
                                                    translation=True)

        i = 0

        for se in episode:
            # print(se)
            # episode index (0:hit1, 1:hit2, 2:bounce_wall, 3:bounce_table)
            # p1 = hit1 -> bounce wall
            # p2 = bounce_wall -> bounce table
            # p3 = bounce_table -> hit2

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
            gaze = np.array([movingAverage(gaze_n[p1_s:p3_e, i], n=2) for i in range(3)]).transpose()

            ball = np.array([interpolate(ball_n[p1_s:p3_e, i]) for i in range(3)]).transpose()
            ball = np.array([movingAverage(ball[:, i], n=2) for i in range(3)]).transpose()

            # if np.sum(np.isnan(ball)):
            #     print("error")

            tobii = np.array([movingAverage(self.p.tobii_segment_T[p1_s:p3_e, i], n=1) for i in range(3)]).transpose()
            tobii_avg = np.array(
                [movingAverage(self.p.tobii_segment_T[p1_s:p3_e, i], n=3) for i in range(3)]).transpose()
            onset_offset_saccade, onset_offset_sp, onset_offset_fix, stream_label = detectSaccade(gaze, ball, tobii,
                                                                                                  tobii_avg)

            # normalize gaze and ball
            gaze_ih = gaze - tobii_avg
            ball_ih = ball - tobii_avg
            win_length = 10
            # do normalize the start and the end of the episode
            p1s, p1e = p1_s - p1_s, p1_e - p1_s
            p2s, p2e = p2_s - p1_s, p2_e - p1_s
            p3s, p3e = p3_s - p1_s, p3_e - p1_s

            if len(onset_offset_saccade) > 0:
                saccade_p1 = onset_offset_saccade[onset_offset_saccade[:, 0] <= p1e]
                saccade_p2 = onset_offset_saccade[
                    (onset_offset_saccade[:, 0] > p2s) & (onset_offset_saccade[:, 0] <= p2e)]
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
                    saccades_phase1, al_cs_p1 = saccadeFeatures(saccade_p1, gaze_ih, ball_ih, win_length=10,
                                                                phase_start=p1s, phase_end=p1e)
                    al_cs_labels.append(al_cs_p1)
                    gaze_event_list[i, 3] = np.sum(al_cs_p1 == 2)  # compute non AL saccade in phase 1
                    if np.sum(al_cs_p1 == 1) == 1:
                        anticipation_phase1_list[i] = saccades_phase1[al_cs_p1 == 1]
                        # if saccades_phase1[al_cs_p1 == 1][0][5] > 100:
                        #     print(se)
                        gaze_event_list[i, 0] = 1

                # print("------------------------Phase2------------------------")
                if len(saccade_p2) > 0:
                    saccades_phase2, al_cs_p2 = saccadeFeatures(saccade_p2, gaze_ih, ball_ih,
                                                                win_length=10, phase_start=p2s, phase_end=p2e,
                                                                phase_2=True)
                    al_cs_labels.append(al_cs_p2)
                    gaze_event_list[i, 4] = np.sum(al_cs_p2 == 2)  # compute non AL saccade in phase 2
                    if np.sum(al_cs_p2 == 1) == 1:

                        anticipation_phase2_list[i] = saccades_phase2[al_cs_p2 == 1]
                        # if saccades_phase2[al_cs_p2 == 1][0][5] > 100:
                        #     print(se)

                        gaze_event_list[i, 1] = 1
                        if len(onset_offset_sp) > 0:
                            fsp_phase3 = fixationSPFeatures(onset_offset_sp, saccade_p2[al_cs_p2 == 1], gaze_ih,
                                                            ball_ih, phase_start=p3s, phase_end=p3e, phase2_start=p2s)
                            fsp_phase3_list[i] = fsp_phase3
                            gaze_event_list[i, 2] = 1 if np.sum(np.isnan(fsp_phase3) == 0) else 0

                # print("------------------------Phase3------------------------")
                if len(saccade_p3) > 0:
                    if saccade_p3[-1, 1] > len(gaze):
                        saccade_p3[-1, 1] = len(gaze) - 1
                    win_length = len(ball) - saccade_p3[-1, 1]
                    saccades_phase3, al_cs_p3 = saccadeFeatures(saccade_p3, gaze_ih, ball_ih,
                                                                win_length=win_length, phase_start=p3s, phase_end=p3e,
                                                                classify_al=False)

                    if np.sum(al_cs_p3 == 1) == 1:
                        anticipation_phase3_list[i] = saccades_phase3[al_cs_p3 == 1]
                    al_cs_labels.append(al_cs_p3)

                # if you want to plot the results
                # al_cs_labels = np.concatenate(al_cs_labels)
                # gazeEventsPlotting(gaze, tobii_avg, ball, onset_offset_saccade, onset_offset_sp, onset_offset_fix,
                #                    stream_label, al_cs_labels)

            i += 1

        features_summary = {
            "saccade_p1": anticipation_phase1_list,
            "saccade_p2": anticipation_phase2_list,
            "saccade_p3": anticipation_phase3_list,
            "gaze_event": gaze_event_list,
            "fsp_phase3": fsp_phase3_list,
        }

        return features_summary

    def detectJointAttention(self, episode, s2, th=10):
        '''
        :param episode: list of episode
        :param s2: compared subject
        :param ball: ball vector
        :param th: threshold for joint attention |s1.gaze - ball| < th
        :return: list
        '''

        # saccades features
        joint_attention_list = np.empty(shape=(len(episode), 4))
        joint_attention_list[:] = np.nan

        ball_n = self.b.ball_t

        gaze_s1_n = self.tobii_reader.local2GlobalGaze(self.p.gaze_point, self.p.tobii_segment_T,
                                                       self.p.tobii_segment_R,
                                                       translation=True)
        gaze_s2_n = self.tobii_reader.local2GlobalGaze(s2.p.gaze_point, s2.p.tobii_segment_T,
                                                       s2.p.tobii_segment_R,
                                                       translation=True)

        i = 0
        for se in episode:
            # print(se)
            # episode index (0:hit1, 1:hit2, 2:bounce_wall, 3:bounce_table)
            # p1 = hit1 -> bounce wall
            # p2 = bounce_wall -> bounce table
            # p3 = bounce_table -> hit2

            # phase 1
            p1_s = se[0]
            p1_e = se[2]

            # phase 2
            p2_s = se[2]
            p2_e = se[3]

            # phase 3
            p3_s = se[3]
            p3_e = se[1]

            # do normalize the start and the end of the episode
            p1s, p1e = p1_s - p1_s, p1_e - p1_s
            p2s, p2e = p2_s - p1_s, p2_e - p1_s
            p3s, p3e = p3_s - p1_s, p3_e - p1_s

            # normalize data
            gaze_s1 = np.array([movingAverage(gaze_s1_n[p1_s:p3_e, i], n=2) for i in range(3)]).transpose()
            gaze_s2 = np.array([movingAverage(gaze_s2_n[p1_s:p3_e, i], n=2) for i in range(3)]).transpose()

            ball = np.array([interpolate(ball_n[p1_s:p3_e, i]) for i in range(3)]).transpose()
            ball = np.array([movingAverage(ball[:, i], n=2) for i in range(3)]).transpose()

            tobii_avg_s1 = np.array(
                [movingAverage(self.p.tobii_segment_T[p1_s:p3_e, i], n=3) for i in range(3)]).transpose()

            tobii_avg_s2 = np.array(
                [movingAverage(s2.p.tobii_segment_T[p1_s:p3_e, i], n=3) for i in range(3)]).transpose()

            # normalize gaze and ball
            gaze_ih_s1 = gaze_s1 - tobii_avg_s1
            ball_ih_s1 = ball - tobii_avg_s1

            gaze_ih_s2 = gaze_s2 - tobii_avg_s2
            ball_ih_s2 = ball - tobii_avg_s2

            joint_attention_feature = jointAttentionFeatures(gaze_ih_s1, gaze_ih_s2, ball_ih_s1, ball_ih_s2, th=th)

            # gaze_movement_sync_p1 = gazeMovementFeatures(gaze_ih_s1[p1s:p1e + 1], gaze_ih_s2[p1s:p1e + 1],
            #                                                ball_ih_s1[p1s:p1e + 1], ball_ih_s2[p1s:p1e + 1])

            # gaze_movement_sync_p2 = gazeMovementFeatures(gaze_ih_s1[p2s:p2e + 1], gaze_ih_s2[p2s:p2e + 1],
            #                                                ball_ih_s1[p2s:p2e + 1], ball_ih_s2[p2s:p2e + 1],)

            gaze_movement_sync_p3 = gazeMovemenSynctFeatures(gaze_ih_s1[p3s:p3e + 1], gaze_ih_s2[p3s:p3e + 1],
                                                           ball_ih_s1[p3s:p3e + 1], ball_ih_s2[p3s:p3e + 1])

            joint_attention_list[i] = joint_attention_feature
            i += 1

        features_summary = {
            "joint_attention": joint_attention_list,

        }

        return features_summary

    def speedACCBeforeImpact(self, episode, win_length=3):

        def computeForceImpact(ball_imp_vel):
            # compute force of impact
            after_imp = np.nanargmax(ball_imp_vel)
            in_bal_im = ball_imp_vel[0]
            af_bal_im = ball_imp_vel[after_imp]
            if after_imp == 0:
                return 0
            return (BALL_MASS * (af_bal_im - in_bal_im)) / (after_imp + 1)

        def computeBallWrist():
            # distance from wrist
            right_wrist = whichShoulder(self.p.racket_segment_T[end_idx], self.p.rwirst_segment_T[end_idx],
                                        self.p.lwirst_segment_T[end_idx])

            if right_wrist:
                wrist = self.p.rwirst_segment_T[end_idx]
            else:
                wrist = self.p.lwirst_segment_T[end_idx]

            ball_on_wrist = np.linalg.norm(wrist - self.b.ball_t[end_idx])

            return ball_on_wrist

        racket_force_list = []
        ball_force_list = []
        rball_ratio_list = []
        ball_on_rack_list = []
        ball_force_imp_list = []
        ball_to_wrist_list = []
        ball_rack_wrist_list = []
        for e in episode:
            # phase 3
            start_idx = e[2]
            end_idx = e[1]
            racket_forward = self.p.racket_segment_T[start_idx:end_idx]
            ball_forward = self.b.ball_t[start_idx:end_idx]
            racket_speed, _, racket_acc = computeVelAcc(racket_forward, num_frames=100)
            ball_speed, _, ball_acc = computeVelAcc(ball_forward, num_frames=100)

            ball_impact = self.b.ball_t[end_idx - 3:end_idx + 2]
            _, ball_imp_vel, _ = computeVelAcc(ball_impact, num_frames=100)

            # compute forces
            # if (np.sum(np.isnan(racket_forward)) != 0) | (np.sum(np.isnan(ball_forward)) != 0):
            #     print("racket error")
            racket_force = (np.nanmean(racket_acc[-win_length:])) * RACKET_MASS
            ball_force = (np.nanmean(ball_acc[-win_length:])) * BALL_MASS

            if np.abs(racket_force) > 0.01:
                racket_force = np.nan
            if np.abs(ball_force) > 0.005:
                ball_force = np.nan
                # print("racket error")
            rball_ratio = racket_force - ball_force
            ball_force_imp = computeForceImpact(ball_imp_vel)
            ball_on_rack = np.min(np.linalg.norm(
                self.p.racket_segment_T[end_idx - 1:end_idx + 1] - self.b.ball_t[end_idx - 1:end_idx + 1], axis=-1))
            # if ball_on_rack > 200:
            #     print("racket error")
            #     print(e)
            ball_to_wrist = computeBallWrist()
            contact_point = self.b.ball_t[end_idx:end_idx + 1]
            incident_angle = computeSegmentAngles(contact_point - self.p.racket_segment_T[end_idx - 2:end_idx - 1],
                                                  contact_point - self.b.ball_t[end_idx - 2:end_idx - 1])[0]
            racket_force_list.append(racket_force)
            ball_force_list.append(ball_force)
            rball_ratio_list.append(incident_angle)
            ball_force_imp_list.append(ball_force_imp)
            ball_on_rack_list.append(ball_on_rack)
            ball_to_wrist_list.append(ball_to_wrist)
            ball_rack_wrist_list.append(np.abs(ball_to_wrist - ball_on_rack))

        features_summary = {
            "im_racket_force": np.expand_dims(np.asarray(racket_force_list), -1),
            "im_ball_force": np.expand_dims(np.asarray(ball_force_list), -1),
            "im_rb_ang_collision": np.expand_dims(np.asarray(rball_ratio_list), -1),
            "im_ball_fimp": np.expand_dims(np.asarray(ball_force_imp_list), -1),
            "im_rb_dist": np.expand_dims(np.asarray(ball_on_rack_list), -1),
            "im_to_wrist_dist": np.expand_dims(np.asarray(ball_to_wrist_list), -1),
            "im_rack_wrist_dist": np.expand_dims(np.asarray(ball_rack_wrist_list), -1),

        }
        return features_summary

    def speedACCForwardSwing(self, episode: np.array, start_fs: np.array, n_window=5) -> dict:

        ball_speed_list = []
        racket_speed_list = []
        ball_acc_list = []
        racket_acc_list = []
        rball_dir_list = []
        rball_dist = []
        for e, fs_idx in zip(episode, start_fs):
            impact_idx = e[1]
            start_idx = e[2] + fs_idx
            if start_idx + n_window < impact_idx:
                sample_forward = start_idx + n_window
            else:
                sample_forward = impact_idx

            if sample_forward - start_idx < n_window:
                fs_idx -= 2
            start_idx = e[2] + fs_idx
            # racket_mag = np.linalg.norm(self.racket_segment_T[fs_idx] - self.racket_segment_T[impact_idx])
            # ball_mag = np.linalg.norm(self.ball_t[fs_idx] - self.ball_t[impact_idx])
            racket_forward = self.p.racket_segment_T[start_idx:sample_forward]
            ball_forward = self.b.ball_t[start_idx:sample_forward]

            racket_speed, _, racket_acc = computeVelAcc(racket_forward, num_frames=100)
            ball_speed, _, ball_acc = computeVelAcc(ball_forward, num_frames=100)

            r_ball_dist = np.linalg.norm(self.p.racket_segment_T[start_idx] - self.b.ball_t[start_idx])
            r_ball_dir = np.nanmean(
                computeVectorsDirection(racket_forward[1:] - racket_forward[0:1], ball_forward[1:] - ball_forward[0:1]))

            # if (np.average(np.isnan(racket_acc)) == 1) | len(racket_acc) == 0:
            #     print("error")
            ball_speed_list.append(np.nanmean(ball_speed))
            racket_speed_list.append(np.nanmean(racket_speed))
            racket_acc_list.append(np.nanmean(racket_acc))
            ball_acc_list.append(np.nanmean(ball_acc))
            rball_dir_list.append(r_ball_dir)
            rball_dist.append(r_ball_dist)

        features_summary = {
            "racket_speed": np.expand_dims(np.asarray(racket_speed_list), -1),
            "racket_acc": np.expand_dims(np.asarray(racket_acc_list), -1),
            "ball_speed": np.expand_dims(np.asarray(ball_speed_list), -1),
            "ball_acc": np.expand_dims(np.asarray(ball_acc_list), -1),
            "rball_dir": np.expand_dims(np.asarray(rball_dir_list), -1),
            "rball_dist": np.expand_dims(np.asarray(rball_dist), -1),

        }

        return features_summary

    def startForwardSwing(self, episode: np.array) -> dict:
        '''
        compute the start time of forward swing
        start of forward swing: total number of frames taken for individuals to hit the ball after it bounces on the wall
        :param episode: episodes
        :return: the list of start time of forward swing
        '''
        dist_racket_wall = np.linalg.norm(self.tw.wall_segment - self.p.racket_segment_T, axis=-1)
        vel_racket = np.abs(dist_racket_wall[1:, ] - dist_racket_wall[:-1, ])
        # vel_racket = np.linalg.norm(self.p.racket_segment_T[1:, ] - self.p.racket_segment_T[:-1, ], axis=-1)
        acc_racket = np.diff(np.pad(vel_racket, (1, 1), 'symmetric'), n=1, axis=0)
        rt_list = []
        rt_list_idx = []
        for e in episode:
            start = e[2]
            end = e[1]

            rt_vel = acc_racket[start:end]
            peaks, _ = find_peaks(rt_vel, distance=10)
            # print()
            #
            # plt.subplot(2, 1, 1)
            # plt.plot(rt_vel)
            # plt.vlines(peaks, np.min(racket_split), np.max(racket_split), colors="red")
            # plt.subplot(2, 1, 2)
            # plt.plot(dist_racket_wall[start:end])
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
                rt_list.append(0)
                rt_list_idx.append(end - start)
            else:
                high_peaks = peaks[-1]
                # rt = (e[1] - e[2]) - high_peaks
                # rt = (e[1] - e[2]) - high_peaks
                rt_list.append(high_peaks)
                rt_list_idx.append(high_peaks)

        features_summary = {
            "avg_start_fs": np.nanmean(np.asarray(rt_list)),
            "std_start_fs": np.nanstd(rt_list),
            "start_fs": np.expand_dims(np.asarray(rt_list_idx), -1)
        }

        return features_summary

    def angleRacketShoulder(self, episodes: np.array, start_fs: np.array) -> dict:
        '''
        :param episodes:
        :return:

        e1 = bounce on the wall
        e2 = bounce on the table
        e3 = start of forward swing
        e4 = ball impact
        '''

        ball_norm = self.b.ball_t - self.tw.wall_segment
        racket_norm = self.p.racket_segment_T - self.tw.wall_segment
        rcollar_norm = self.p.rcollar_segment_T - self.tw.wall_segment
        lcollar_norm = self.p.lcollar_segment_T - self.tw.wall_segment
        angles_list = []

        for e, fs in zip(episodes, start_fs):
            b1_idx = e[2]
            b2_idx = e[3]
            b4_idx = e[1]

            right_shoulder = whichShoulder(self.p.racket_segment_T[b1_idx], self.p.rwirst_segment_T[b1_idx],
                                           self.p.lwirst_segment_T[b1_idx])

            if right_shoulder:
                collar = rcollar_norm
            else:
                collar = lcollar_norm

            # e1_angle = np.std(computeSegmentAngles(ball_norm[b1_idx:b2_idx ], racket_norm[b1_idx:b2_idx]), keepdims=True)
            # e2_angle = np.std(computeSegmentAngles(ball_norm[b2_idx:fs], racket_norm[b2_idx:fs]), keepdims=True)
            # e3_angle = np.std(computeSegmentAngles(ball_norm[fs:b4_idx], racket_norm[fs:b4_idx]), keepdims=True)
            # e4_angle = computeSegmentAngles(racket_norm[b4_idx:b4_idx + 1], collar[b4_idx:b4_idx + 1])
            e1_angle = computeSegmentAngles(ball_norm[b1_idx:b1_idx + 1], racket_norm[b1_idx:b1_idx + 1])
            e2_angle = computeSegmentAngles(ball_norm[b2_idx:b2_idx + 1], racket_norm[b2_idx:b2_idx + 1])
            e3_angle = computeSegmentAngles(ball_norm[fs:fs + 1], racket_norm[fs:fs + 1])
            e4_angle = computeSegmentAngles(racket_norm[b4_idx:b4_idx + 1], collar[b4_idx:b4_idx + 1])
            angles_list.append([e1_angle, e2_angle, e3_angle, e4_angle])

        features_summary = {
            "events_twoseg_angles": np.asarray(angles_list)[:, :, 0]
        }

        return features_summary
