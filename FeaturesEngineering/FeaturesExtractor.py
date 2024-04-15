import numpy as np
from Utils.Lib import wienerFilter, movingAverage, savgolFilter, cartesianToSpher
from FeaturesEngineering.GazeEvent import saccadeFeatures, detectSaccade, fixationSPFeatures, jointAttentionFeatures, gazeMovemenCoorientationtFeatures
from Utils.DataReader import TobiiReader
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from Utils.Interpolate import interpolate
from FeaturesEngineering.Plotting import gazeEventsPlotting
from FeaturesEngineering.FeaturesLib import computeVectorsDirection, whichShoulder, computeSegmentAngles, computeVelAcc, computeMultiDimSim, computeKineticEnergy
from Analysis.Conf import RACKET_MASS, BALL_MASS
from FeaturesEngineering.SubjectObject import Subject, Ball, TableWall


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

    def detectGazeEvents(self, episode, th_al_p1=10, th_al_p2=25, th_angle_p=15, normalize=True, fill_in=False, iam_idx=0, double=False) -> dict:
        '''
        Detect the onset and offset of saccade and pursuit in 3 phases
        p1 = hit1 -> bounce wall
        p2 = bounce_wall -> bounce table
        p3 = bounce_table -> hit2

        In case of double
        p1 = hitter -> bounce wall
        p2 = bounce wall -> bounce table
        p3 = bounce table -> receiver

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
        '''
        Anticipation features:
        0: onset
        1: precision of AL
        2: magnitude of AL
        '''
        anticipation_phase1_list = np.empty(shape=(len(episode), 3))

        anticipation_phase2_list = np.empty(shape=(len(episode), 3))

        anticipation_phase3_list = np.empty(shape=(len(episode), 3))

        '''
        gaze event list
        0: whether there is AL in Phase 1 or not
        1: whether there is AL in Phase 2 or not
        2: whether there is FX in Phase 3 or not 
        3: number of corrective saccade in Phase 1
        4: number of corrective saccade in Phase 2
        5: number of catch up saccade during pursuit
        '''
        gaze_event_list = np.zeros(shape=(len(episode), 6))

        # attended fixation and sp features
        fsp_phase3_list = np.empty(shape=(len(episode), 3))

        th_pursuit = 10 # default of pursuit threshold
        if fill_in:
            anticipation_phase1_list[:] = 0
            anticipation_phase2_list[:] = 0
            anticipation_phase3_list[:] = 0
            fsp_phase3_list[:] = 0
        else:
            anticipation_phase1_list[:] = np.nan
            anticipation_phase2_list[:] = np.nan
            anticipation_phase3_list[:] = np.nan
            fsp_phase3_list[:] = np.nan


        ball_n = self.b.ball_t

        gaze_n = self.tobii_reader.local2GlobalGaze(self.p.gaze_point, self.p.tobii_segment_T, self.p.tobii_segment_R,
                                                    translation=True)

        i = 0

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

            # detect saccade
            gaze = np.array([movingAverage(gaze_n[p1_s:p3_e, i], n=2) for i in range(3)]).transpose()

            ball = np.array([interpolate(ball_n[p1_s:p3_e, i]) for i in range(3)]).transpose()
            ball = np.array([movingAverage(ball[:, i], n=2) for i in range(3)]).transpose()

            # if np.sum(np.isnan(ball)):
            #     print("error")

            tobii = np.array([movingAverage(self.p.tobii_segment_T[p1_s:p3_e, i], n=1) for i in range(3)]).transpose()
            tobii_avg = np.array(
                [movingAverage(self.p.tobii_segment_T[p1_s:p3_e, i], n=3) for i in range(3)]).transpose()


            # if double:
            #     if se[6] != iam_idx:
            #         th_pursuit = 15 # if i am not the receiver, decerease the threshold

            onset_offset_saccade, onset_offset_sp, onset_offset_fix, stream_label = detectSaccade(gaze, ball, tobii,
                                                                                                  tobii_avg, th_pursuit=th_pursuit)

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

                al_cs_labels = []
                # Phase 1
                if len(saccade_p1) > 0:
                    saccades_phase1, al_cs_p1 = saccadeFeatures(saccade_p1, gaze_ih, ball_ih, win_length=10,
                                                                phase_start=p1s, phase_end=p1e)
                    al_cs_labels.append(al_cs_p1)
                    gaze_event_list[i, 3] = np.sum(al_cs_p1 == 2)  # compute non AL saccade after AL in phase 1
                    if np.sum(al_cs_p1 == 1) == 1:
                        anticipation_phase1_list[i] = saccades_phase1[al_cs_p1 == 1]
                        # if saccades_phase1[al_cs_p1 == 1][0][5] > 100:
                        #     print(se)
                        gaze_event_list[i, 0] = 1

                # Phase 2
                if len(saccade_p2) > 0:
                    saccades_phase2, al_cs_p2 = saccadeFeatures(saccade_p2, gaze_ih, ball_ih,
                                                                win_length=10, phase_start=p2s, phase_end=p2e,
                                                                phase_2=True)
                    al_cs_labels.append(al_cs_p2)
                    gaze_event_list[i, 4] = np.sum(al_cs_p2 == 2) # compute non AL saccade after AL in phase 2
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

                # Phase 3
                if len(saccade_p3) > 0:

                    if saccade_p3[-1, 1] > len(gaze):
                        saccade_p3[-1, 1] = len(gaze) - 1
                    win_length = len(ball) - saccade_p3[-1, 1]
                    saccades_phase3, al_cs_p3 = saccadeFeatures(saccade_p3, gaze_ih, ball_ih,
                                                                win_length=win_length, phase_start=p3s, phase_end=p3e,
                                                                classify_al=False)

                    if gaze_event_list[i, 2] == 1:
                        gaze_event_list[i, 5] = np.sum(al_cs_p3 != 1)
                    if np.sum(al_cs_p3 == 1) == 1:
                        anticipation_phase3_list[i] = saccades_phase3[al_cs_p3 == 1]
                    al_cs_labels.append(al_cs_p3)

                # if you want to plot the results
                # print(p1e)
                # print(p2e)
                # print(gaze_event_list[i, 5])
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


    def detectGazeCoorientation(self, episode, s2, b, th=20, fill_in=False):
        '''
          :param episode: list of episode
          :param s2: compared subject
          :param ball: ball vector
          :param th: threshold for joint attention |s1.gaze - ball| < th
          :return: list
          '''

        # saccades features
        joint_attention_list_p1 = np.empty(shape=(len(episode), 4))
        joint_attention_list_p23 = np.empty(shape=(len(episode), 4))
        gaze_coor_p1 = np.empty(shape=(len(episode), 13))
        gaze_coor_p23 = np.empty(shape=(len(episode), 13))

        if fill_in:
            joint_attention_list_p1[:] = 0
            joint_attention_list_p23[:] = 0
            gaze_coor_p1[:] = 0
            gaze_coor_p23[:] = 0
        else:
            joint_attention_list_p1[:] = np.nan
            joint_attention_list_p23[:] = np.nan
            gaze_coor_p1[:] = np.nan
            gaze_coor_p23[:] = np.nan

        ball_n = b.ball_t

        gaze_s1_n = self.tobii_reader.local2GlobalGaze(self.p.gaze_point, self.p.tobii_segment_T,
                                                       self.p.tobii_segment_R,
                                                       translation=True)
        gaze_s2_n = self.tobii_reader.local2GlobalGaze(s2.p.gaze_point, s2.p.tobii_segment_T,
                                                       s2.p.tobii_segment_R,
                                                       translation=True)

        i = 0
        avg_phase_all = []
        surr_avg_phase_all = []
        for se in episode:
            # print(se[1] - se[0])
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


            # joint attention P2+3
            # print(p2s)
            # print(p2e)
            gazeMovemenCoorientationtFeatures(gaze_ih_s1[p2s:p3e], gaze_ih_s2[p2s:p3e],
                                                                      ball_ih_s1[p2s:p3e], ball_ih_s2[p2s:p3e],
                                                                      full=True)

            # avg_phase_all.append(avg_phase)
            # surr_avg_phase_all.append(surr_phase)

        return avg_phase_all, surr_avg_phase_all





    def detectJointAttention(self, episode, s2, b, th=10, fill_in=False):
        '''
        :param episode: list of episode
        :param s2: compared subject
        :param ball: ball vector
        :param th: threshold for joint attention |s1.gaze - ball| < th
        :return: list
        '''

        # saccades features
        joint_attention_list = np.empty(shape=(len(episode), 1))



        if fill_in:
            joint_attention_list[:] = 0

        else:
            joint_attention_list[:] = np.nan


        ball_n = b.ball_t

        gaze_s1_n = self.tobii_reader.local2GlobalGaze(self.p.gaze_point, self.p.tobii_segment_T,
                                                       self.p.tobii_segment_R,
                                                       translation=True)
        gaze_s2_n = self.tobii_reader.local2GlobalGaze(s2.p.gaze_point, s2.p.tobii_segment_T,
                                                       s2.p.tobii_segment_R,
                                                       translation=True)

        i = 0
        for se in episode:
            # print(se[1] - se[0])
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




            # joint attention P1
            joint_attention_feature = jointAttentionFeatures(gaze_ih_s1[p2s:p3e + 1], gaze_ih_s2[p2s:p3e + 1],
                                                                ball_ih_s1[p2s:p3e + 1], ball_ih_s2[p2s:p3e + 1], th=th)






            joint_attention_list[i] = joint_attention_feature

            i += 1

        features_summary = {

            "joint_attention": joint_attention_list,


        }

        return features_summary

    def speedACCBeforeImpact(self, episode, win_length=3):
        '''
        extract features before impact
        :param episode: list of episode
        :param win_length: the number of frames before the impact
        :return:
        im_racket_force: kinetic energy of racket before impact
        im_ball_force: kinectic energy of ball before impact
        im_rb_ang_collision: incident angle during the impact
        im_ball_fimp: the force during the impact
        im_rb_dist: the position of ball relative to the racket segment at the impact
        im_to_wrist_dist: the position of the ball relative to the wrist
        im_rack_wrist_dist: the position of the ball relative to the racket and the wrist
        '''

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
            racket_speed, _, racket_acc = computeVelAcc(racket_forward, fps=100)
            ball_speed, _, ball_acc = computeVelAcc(ball_forward, fps=100)


            # compute the force during the impact
            ball_impact = self.b.ball_t[end_idx - win_length:end_idx + 2]
            _, ball_imp_vel, _ = computeVelAcc(ball_impact, fps=100)
            ball_force_imp = computeForceImpact(ball_imp_vel)

            # compute kinematic energy before the impact
            racket_force = computeKineticEnergy(racket_forward, n_window=win_length, mass=RACKET_MASS, fps=100)
            ball_force = computeKineticEnergy(ball_forward, n_window=win_length, mass=BALL_MASS, fps=100)


            if np.abs(racket_force) > 0.1:
                racket_force = np.nan
            if np.abs(ball_force) > 0.005:
                ball_force = np.nan
                # print("racket error")


            # compute the position of the ball relative to the racket segment
            ball_on_rack = np.min(np.linalg.norm(
                self.p.racket_segment_T[end_idx - 1:end_idx + 1] - self.b.ball_t[end_idx - 1:end_idx + 1], axis=-1))

            # compute the position of the ball relative to wrist
            ball_to_wrist = computeBallWrist()



            # compute the incident angle with reference to the ball position during impact
            contact_point = self.b.ball_t[end_idx:end_idx + 1]
            incident_angle = computeSegmentAngles(self.p.racket_segment_T[end_idx - 2:end_idx - 1] - contact_point,
                                                  self.b.ball_t[end_idx - 2:end_idx - 1] - contact_point)[0]



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
        '''
        extract features when performing forward swing
        :param episode:
        :param start_fs:
        :param n_window:
        :return:
        racket speed
        racket velocity
        ball speed
        ball velocity
        rball_dir: direction of ball and racket from the start of forward swing and impact
        rball_dist: the distance between the ball and racket when the forward swing start
        '''

        ball_speed_list = []
        racket_speed_list = []
        ball_acc_list = []
        racket_acc_list = []
        rball_dir_list = []
        rball_dist = []
        for e, fs_idx in zip(episode, start_fs):
            impact_idx = e[1]
            # start forward is wall bounce + onset of the forward swing
            start_idx = e[2] + fs_idx
            sample_forward = impact_idx

            # if there is too less sample, start of forward swing - 2
            if sample_forward - start_idx < n_window:
                fs_idx -= (n_window - (sample_forward - start_idx))
                start_idx = e[2] + fs_idx


            racket_forward = self.p.racket_segment_T[start_idx:sample_forward]
            ball_forward = self.b.ball_t[start_idx:sample_forward]

            racket_speed, _, racket_acc = computeVelAcc(racket_forward, fps=100)
            ball_speed, _, ball_acc = computeVelAcc(ball_forward, fps=100)

            r_ball_dist = np.linalg.norm(self.p.racket_segment_T[start_idx] - self.b.ball_t[start_idx])

            # direction of the racket relative to direction of the ball from the start of the forward swing until the impact
            r_ball_dir = np.nanmean(
                computeVectorsDirection(racket_forward[-2:] - racket_forward[0:1], ball_forward[-2:] - ball_forward[0:1]))


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
        acc_racket = np.diff(np.pad(vel_racket, (1, 1), 'symmetric'), n=1, axis=0)
        rt_list = []
        rt_list_idx = []
        for e in episode:
            start = e[2]
            end = e[1]

            rt_vel = acc_racket[start:end]

            rt_vel = movingAverage(rt_vel, 3) # perform moving average to remove noise

            peaks, _ = find_peaks(rt_vel, distance=10)

            if len(peaks) == 0:

                rt_list.append(0)
                rt_list_idx.append(end - start)
            else:
                high_peaks = peaks[-1]
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

    def extractSpeedForwardSwing(self, start, end):
        racket_seg = self.p.racket_segment_T[int(start):int(end)]
        vel = np.linalg.norm(racket_seg[1:] - racket_seg[:-1], axis=-1)
        acc = np.diff(vel, n=1, axis=0)
        return acc



    def racketMovementSimilarity(self, start, end, ref):
        my_vel = self.extractSpeedForwardSwing(start, end)
        dtw_dist, lcc_dist = computeMultiDimSim(ref, my_vel)
        features_summary = {
            "dtw_dist":dtw_dist,
            "lcc_dist": lcc_dist,

        }
        return features_summary

    def distanceDoubleHit(self, episode, ref, xy=True, n_window=10):
        '''
        Compute the relative angle between me and my partner before the impact

        Only consider x and y
        :param episode:
        :param ref:
        :return:
        '''
        features = np.empty((len(episode), 1))
        i=0
        for e in episode:

            impact_idx = e[1]
            my_position = self.p.lower_back_segment_T [impact_idx-n_window:impact_idx] - self.tw.table_segment
            partner_position = ref.p.lower_back_segment_T[impact_idx-n_window:impact_idx] - self.tw.table_segment

            if xy:
                my_position = my_position[:, [0, 1]]
                partner_position = partner_position[:, [0, 1]]
            # features[i] = np.average(np.abs(np.diff(computeSegmentAngles(my_position, partner_position), axis=0)))
            features[i] = np.average(np.abs(np.diff(np.linalg.norm(my_position - partner_position, axis=-1), axis=0)))
            i +=1

        return features

    def bouncingPoint(self, all_episode, episode, ref=None):
        '''
        Compute the relative position of bouncing position on the wall to the top wall  in X and Z axis
        return x, z point of bouncing points
        :param episode: list of all episodes
        :param episode: list of episodes
        :param centro: the reference can be centroid of the wall of the upper position
        :return:
        relative distance between the bouncing position and the reference
        '''

        #compute centroid

        all_bouncing_idx = all_episode[:, 2]
        all_bouncing_position = self.b.ball_t[all_bouncing_idx]


        if ref is None:
            ref = np.mean(all_bouncing_position, axis=0, keepdims=True)
        else:
            ref = ref

        bouncing_idx = episode[:, 2]
        bouncing_position = self.b.ball_t[bouncing_idx]


        ref = ref[bouncing_idx]
        myself = self.p.lower_back_segment_T[bouncing_idx]

        a = np.linalg.norm(ref[:, [0, 2]] - bouncing_position[:, [0, 2]], axis=-1 , keepdims=True)
        b =  np.linalg.norm(myself[:, [0, 2]] - bouncing_position[:, [0, 2]], axis=-1 , keepdims=True)

        bouncing_position = a / b
        features_summary = {
            "bouncing_position": bouncing_position
        }
        return features_summary


    def impactPoint(self, episode, ref=None):
        '''
        How much the racket move to compensate error?
        Compute the relative distnace between racket and individuals' lower back at the impact time. Compensation often requires individuals to stretch their body
        :param episode: list of all episodes
        :param episode: list of episodes
        :return:
        relative distance between the racket position and the lower back
        '''


        impact_idx = episode[:, 1]
        racket_impact_position = self.p.racket_segment_T[impact_idx]

        table_segment = self.p.lower_back_segment_T[impact_idx]

        root_impact_position = self.p.root_segment_T[impact_idx]

        r_humerus_impact_position = self.p.rhummer_segment_T[impact_idx] - racket_impact_position

        l_humerus_impact_position = self.p.lhummer_segment_T[impact_idx] - racket_impact_position



        if self.p.hand == "R":
            b = np.linalg.norm(racket_impact_position - l_humerus_impact_position, axis=-1, keepdims=True)
        else:
            b = np.linalg.norm(racket_impact_position - r_humerus_impact_position, axis=-1, keepdims=True)


        dist_to_centro = b

        features_summary = {
            "impact_position": dist_to_centro
        }
        return features_summary



