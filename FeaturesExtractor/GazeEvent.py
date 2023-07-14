import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy import signal
from Utils.Lib import movingAverage
from scipy.ndimage import label
from FeaturesExtractor.FeaturesLib import computeVectorsDirection, computeVelAccV2, computeSegmentAngles


def detectSaccade(gaze: np.array, ball: np.array, tobii:np.array,  tobii_avg:np.array):
    def contstructStreamLabel(fixation, sop, saccade):
        # 1 fixation
        # 2 smooth pursuit
        # 3 saccade
        label = np.ones(len(gaze)).astype(int)

        for on, off in fixation:
            label[on:off] = 1
        for on, off in sop:
            label[on:off] = 2
        for on, off in saccade:
            label[on:off] = 3


        return label


    vel_gaze, vel_norm_gaze, acc_gaze = computeVelAccV2(gaze - tobii_avg, normalize=True)
    vel_gaze_h, vel_norm_gaze_h, acc_gaze_h = computeVelAccV2(gaze - tobii,  normalize=False)
    vel_ball, vel_norm_ball, acc_ball = computeVelAccV2(ball - tobii, normalize=False)

    N_label = len(gaze)
    # detect saccade
    labels_saccade = np.zeros(N_label)
    labels_saccade[(vel_norm_gaze > 40) & (acc_gaze >= 15) ] = 1
    onset_offset_saccade = detectOnsetOffset(labels_saccade == 1, type="saccade")

    if len(onset_offset_saccade) > 0:
        onset_offset_saccade[:, 0] = onset_offset_saccade[:, 0] - 1
        onset_offset_saccade[:, 1] = onset_offset_saccade[:, 1] + 2

    # detect pursuit
    labels_sp = np.zeros(N_label)
    vel_ratio = vel_gaze_h / (vel_ball + 1e-15)
    ball_gaze_dist = computeSegmentAngles(ball - tobii, gaze - tobii)
    labels_sp[(ball_gaze_dist<=10) & ((vel_gaze_h >30) & (vel_ratio >= 0.3) & (vel_ratio <= 2.0))] = 1  # pursuit gain
    onset_offset_sp = detectOnsetOffset(labels_sp == 1, type="sp")
    labels_sp[labels_saccade==1] = 0
    onset_offset_sp = groupPursuit(onset_offset_sp)

    # detect fixation
    labels_fix = np.ones(N_label)
    for on, off in onset_offset_saccade:
        labels_fix[on:off+1] = 0
    for on, off in onset_offset_sp:
        labels_fix[on:off + 1] = 0
    onset_offset_fix = detectOnsetOffset(labels_fix == 1, type="fix")
    onset_offset_fix = groupingFixation(gaze, onset_offset_fix)

    stream_label = contstructStreamLabel(onset_offset_fix, onset_offset_sp, onset_offset_saccade)
    # plt.subplot(2, 1, 1)
    # plt.plot(vel_norm_gaze)
    # plt.subplot(2, 1, 2)
    # plt.plot(acc_gaze)
    #
    # plt.show()
    return onset_offset_saccade, onset_offset_sp, onset_offset_fix, stream_label









def saccadeFeatures(eye_event: np.array, gaze: np.array, ball: np.array):
    dist_angle = np.sqrt(np.sum(np.square(gaze - ball), -1))
    onset_offset = detectOnsetOffset(eye_event == 1)
    dirs = computeVectorsDirection(gaze, ball)

    saccades = []
    if len(onset_offset) > 0:
        for on, off in onset_offset:
            # print(on)
            # extract saccade features
            if (np.abs(on) > 10000) & (np.abs(on) > 10000):
                print("Test")

            if ((off - on) > 0):
                saccade_mag = (np.sqrt(np.nansum(np.square(gaze[off] - gaze[on]))) / ((off - on) + 1e-15)) * 100
                saccade_vel = np.linalg.norm(np.diff(gaze[on - 1:on + 1], axis=0)) * 100
                saccade_ang = np.nanmean(dist_angle[on:off])

                # compute med-diff and mean-diff
                med_diff = []
                mean_diff = []
                win_length = 10  # 1 = 10 ms
                for j in [on, off]:
                    bef_idx = (j - win_length) if (j - win_length) >= 0 else 0
                    before_sample = gaze[bef_idx:j]  # 10 = 100 ms
                    after_sample = gaze[j + 1:j + win_length]
                    if (len(before_sample) > 0) & (len(after_sample) > 0):
                        med_diff.append(
                            np.linalg.norm(np.nanmedian(before_sample, axis=0) - np.nanmedian(after_sample, axis=0)))
                        mean_diff.append(
                            np.linalg.norm(np.nanmean(before_sample, axis=0) - np.nanmean(after_sample, axis=0)))
                saccades.append([on * 10, off * 10, (off - on) * 10, saccade_vel, saccade_ang, np.average(med_diff),
                                 np.average(mean_diff)])

    return np.asarray(saccades)


def detectOnsetOffset(v, type="saccade"):
    v = v.astype(np.int)
    output = []
    valley_groups, num_groups = label(v == 1)
    if num_groups > 0:
        for i in np.unique(valley_groups)[1:]:
            valley_group = np.argwhere(valley_groups == i)

            on = np.min(valley_group)
            off = np.max(valley_group)
            if type == "sp":
                th_duration = 1
            elif type == "fix":
                th_duration = 0
            else:
                th_duration = 0
            if (off - on) > th_duration:
                output.append([on, off])

    output = np.asarray(output).astype(int)

    return output


def groupingFixation(gaze: np.array, onset_offset:np.array):
    if (len(onset_offset) > 1):
        new_onset_offset = []


        # print(onset_offset)
        for i in range(len(onset_offset)):

            c_off = onset_offset[i, 1]  # current offset
            onset = onset_offset[i, 0]
            offset = onset_offset[i, 1]
            for j in range(i+1, len(onset_offset)):

                n_on = onset_offset[j, 0]  # current on
                gaze_dist = computeSegmentAngles(gaze[n_on:n_on+1], gaze[c_off:c_off+1])
                if ((n_on - c_off) < 5) & (gaze_dist <= 3):
                    c_off = onset_offset[j, 1]
                    offset = onset_offset[j, 1]
                    onset_offset[j] = onset_offset[j]  * 0

            new_onset_offset.append([onset, offset])

        new_onset_offset = np.asarray(new_onset_offset).astype(int)
        # delete merged elements
        new_onset_offset = new_onset_offset[new_onset_offset[:, 1] != 0]
        return new_onset_offset
    else:
        return onset_offset


def groupPursuit(onset_offset:np.array):
    if (len(onset_offset) > 1):
        new_onset_offset = []

        # print(onset_offset)
        for i in range(len(onset_offset)):

            c_off = onset_offset[i, 1]  # current offset
            onset = onset_offset[i, 0]
            offset = onset_offset[i, 1]
            for j in range(i + 1, len(onset_offset)):

                n_on = onset_offset[j, 0]  # current on
                if n_on - c_off < 3:
                    c_off = onset_offset[j, 1]
                    offset = onset_offset[j, 1]
                    onset_offset[j] = onset_offset[j] * 0

            new_onset_offset.append([onset, offset])

        new_onset_offset = np.asarray(new_onset_offset).astype(int)
        # delete merged elements
        new_onset_offset = new_onset_offset[new_onset_offset[:, 1] != 0]
        return new_onset_offset
    else:
        return onset_offset


def detectALPhase1(eye_event: np.array, az: np.array, elv: np.array, th=10) -> np.array:
    '''
    :param eye_event: eye event of phase 1 from Tobii
    :param gaze: azimuth (ball-gaze)
    :param ball: elevation (ball-gaze)
    :param th: degree/frame (saccade offset - ball trajectory)
    :return: (angle, onset, offset)
    '''

    dist_angle = np.sqrt(np.square(az) + np.square(elv))
    onset_offset = detectOnsetOffset(eye_event == 1)

    al_angle = 1e+4
    al_offset = 0
    al_onset = 0

    if len(onset_offset) > 0:
        for on, off in onset_offset:
            # angle between ball and gaze after saccade offset
            min_angle = np.min(dist_angle[off:])
            if min_angle <= th:
                al_offset = off
                al_onset = on
                al_angle = min_angle

    return np.array([al_angle, al_onset, al_offset])


def detectALPhase2(eye_event2: np.array, eye_event3: np.array, gaze: np.array, ball: np.array, th: float = 25,
                   th_angle_p=25) -> np.array:
    '''
    :param eye_event2: eye_event phase 2
    :param eye_event3: eye_event phase 3
    :param gaze: gaze in polar coordinates
    :param ball: ball in polar coordinates
    :param th: degree/frame (saccade offset - ball trajectory) of AL
    :param th_angle_p: degree/frame (saccade offset - ball trajectory) of pursuit
    :return:
    '''

    # anticipation look
    al_angle = 1e+4
    al_magnitude = 1e+4
    al_offset = 0
    al_onset = 0

    # pursuit
    p_avg_angle = 1e+4
    p_on = 0
    p_off = 0

    onset_offset_saccade = detectOnsetOffset(eye_event2 == 1)

    ps2_end = len(eye_event2)
    ps3_start = ps2_end
    dist_angle = np.sqrt(np.sum(np.square(gaze - ball), -1))  # use azimuth and elevation
    # dist_angle = np.sqrt(np.square(gaze[:, 0] - ball[:, 0]))  # use azimuth only
    # detect pursuit
    dist_angle3 = dist_angle[ps3_start:]
    pursuit_events_idx = detectOnsetOffset((eye_event3 == 2) & (dist_angle3 <= th_angle_p))

    if (len(onset_offset_saccade) > 0) & (len(pursuit_events_idx) > 0):
        first_pursuit = pursuit_events_idx[0]
        last_pursuit = pursuit_events_idx[-1]
        ball_pursuit = ball[first_pursuit[0], 0]
        for on, off in onset_offset_saccade:
            gaze_saccade = gaze[off, 0]
            dist_sp = np.sqrt(np.sum(np.square(gaze_saccade - ball_pursuit)))
            # print(dist_sp)
            if dist_sp <= th:
                al_angle = dist_sp
                al_onset = on
                al_offset = off
                al_magnitude = np.sqrt(np.sum(np.square(gaze[off] - gaze[on]))) / ((off - on) + 1e-5)

    if len(pursuit_events_idx) > 0:
        first_pursuit = pursuit_events_idx[0]
        last_pursuit = pursuit_events_idx[-1]
        p_on = first_pursuit[0]
        p_off = last_pursuit[1]
        p_avg_angle_list = []
        for p_on, p_off in pursuit_events_idx:
            p_avg_angle_list.append(dist_angle3[p_on:p_off])

        p_avg_angle = np.average(np.concatenate(p_avg_angle_list))
    else:
        if np.average((eye_event3 == 2) & (dist_angle3 <= th_angle_p)) == 1:
            p_on = 0
            p_off = len(dist_angle3) - 1
            p_avg_angle = np.average(dist_angle3)
        # else:
        #     plt.plot(dist_angle3)
        #     plt.show()

    return np.array([al_angle, al_magnitude, al_onset, al_offset]), np.array([p_avg_angle, p_on, p_off])
