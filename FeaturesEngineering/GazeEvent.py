import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy import signal
from Utils.Lib import movingAverage
from scipy.ndimage import label
from FeaturesEngineering.FeaturesLib import computeVectorsDirection, computeVelAccV2, computeSegmentAngles
from Utils.Lib import cartesianToSpher


def detectSaccade(gaze: np.array, ball: np.array, tobii:np.array,  tobii_avg:np.array):
    '''
    :param gaze: gaze in the world
    :param ball: ball in head vector
    :param tobii: tobii in the world
    :param tobii_avg: tobii with windowed average
    :return:
    - onset and offset saccade
    - onset and offset smooth pursuit
    - onset and offset fixation
    - stream label of gaze event (1: fixation, 2: smooth pursuit, 3: saccade)
    '''
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
    vel_head, vel_norm_head, acc_head =  computeVelAccV2(tobii, normalize=True)

    N_label = len(gaze)
    # detect saccade
    labels_saccade = np.zeros(N_label)
    labels_saccade[(vel_norm_gaze >= 40) & (acc_gaze >= 15) ] = 1
    onset_offset_saccade = detectOnsetOffset(labels_saccade == 1, type="saccade")

    if len(onset_offset_saccade) > 0:
        onset_offset_saccade[:, 0] = onset_offset_saccade[:, 0] -1
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


def classifyALCS(saccade_onset_offset: np.array, gaze: np.array,  gaze_view:np.array, ball_view: np.array) -> np.array:
    '''
    :param saccade_onset_offset:onset and offset episodes of the saccades
    :param gaze: gaze-in-the-world
    :param gaze_view: gaze in the visual field
    :param ball_view: ball in the visual field
    label= (1: anticipatory look, 2: correction saccade)
    :return: labels of each saccade (0: normal saccade, 1: anticipatory look, 2: correction saccade)

    normal saccade: the saccade direction and ball direction differ (before anticipatory)
    anticipatory look: the saccade moves to the same direction as the ball
    correction saccade: the saccade occurs after anticipatory look (after anticipatory)
    '''

    def detectAL(g, b):
        ball_gaze_dir = computeVectorsDirection(g[1:] - g[0:1], b[1:] - b[0:1])

        return ball_gaze_dir[0]

    all_candidates = []
    all_gm = []
    for on, off in saccade_onset_offset:
        idx = on
        gm = computeSegmentAngles(np.expand_dims(gaze[on], 0), np.expand_dims(gaze[off], 0))[0]
        is_al = detectAL(gaze_view[[idx, idx + 2]], ball_view[[idx, idx + 2]])
        all_candidates.append(is_al)
        all_gm.append(gm)
    all_candidates = np.array(all_candidates)
    all_gm = np.array(all_gm)


    # print(all_gm)
    labels = np.zeros_like(all_gm)
    al_idx = np.argmax(all_gm * (all_candidates >=0))

    # if len(np.argwhere(all_candidates[:al_idx+1] >= 0)) == 1:
    #     labels[np.argwhere(all_candidates[:al_idx+1] >= 0)] = 1

    labels[al_idx] = 1
    labels[al_idx+1:] = 2

    # print(all_candidates)
    # print(all_gm)
    # print(labels)

    return labels

def saccadeFeatures(onset_offset: np.array, gaze: np.array, ball: np.array, win_length:int = 10, phase_start=0, phase_end=100, classify_al=True, phase_2=False) -> tuple:
    '''
    Onset                       =on
    Offet                       =off
    Duration                    =dn
    Mean distance angle         =mda
    Mean difference             =md
    Min difference              =mid
    Max difference              =mad
    Mean magnitude              =mm
    Sum magnitude               =sm
    Global magnitude            =gm

    :param onset_offset: onset and offset episodes
    :param gaze: gaze in the world
    :param ball: ball in the world
    :return: concatenate features

    One should note that we use gaze and ball information in the world space and visual-field space
    When you should use gaze-in-the-world
    - computing velocity or acceleration
    when you should use gaze-in-the-head (visual field)
    - when computing angle between the ball and the gaze
    '''



    _, gaze_az, gaze_elv = cartesianToSpher(vector=gaze, swap=False)
    _, ball_az, ball_elv = cartesianToSpher(vector=ball, swap=False)

    # when computing the angle between the gaze and the ball, both of them shuld be converted into visual field
    gaze_view = np.vstack([gaze_az, gaze_elv]).transpose()
    ball_view = np.vstack([ball_az, ball_elv]).transpose()

    dist_angle = np.linalg.norm(gaze_view - ball_view, axis=-1)


    features = []
    if len(onset_offset) > 0:
        for on, off in onset_offset:
            # print(on)
            # extract saccade features

            if ((off - on) > 0):
                gaze_saccade = gaze[on:off+1]
                # compute magnitude
                vel_gaze, vel_norm_gaze, acc_gaze = computeVelAccV2(gaze_saccade, normalize=False)
                sm = np.sum(vel_gaze[:-1])
                mm = np.average(vel_gaze)

                gm = computeSegmentAngles(np.expand_dims(gaze[on], 0), np.expand_dims(gaze[off], 0))[0]

                mda = np.nanmean(dist_angle[on:off])

                # compute med-diff and mean-diff
                ball_after_offset = ball_view[off:off+win_length]
                gaze_at_offset = np.ones_like(ball_after_offset) * gaze_view[off]
                # after_ofset_angle = computeSegmentAngles(np.ones_like(ball_after_offset) * gaze_view[off], ball_after_offset)
                after_ofset_angle = np.linalg.norm(gaze_at_offset- ball_after_offset, axis=-1)


                on_event = (phase_end - on) * 10
                off_event = (phase_end - off) * 10

                # if phase_2:
                #     on_event = (on - phase_start) * 10
                #     off_event = (off - phase_start) * 10
                # else:
                #     on_event = (phase_end - on) * 10
                #     off_event = (phase_end - off) * 10

                # concatenate saccades features
                try:
                    features.append([
                                     on_event,
                                     off_event,
                                     (off - on) * 10,
                                     mda,
                                     np.nanmean(after_ofset_angle),
                                     np.nanmin(after_ofset_angle),
                                     np.nanmax(after_ofset_angle),
                                     mm,
                                     sm,
                                     gm
                                     ])
                except:
                    print("error")

    # remove nan features
    features = np.asarray(features)
    features_clean = features[np.sum(np.isnan((features)), -1) == 0]

    # classify whether a saccade is anticipitary look or correction saccade
    if classify_al:
        al_cs = classifyALCS(onset_offset, gaze, gaze_view, ball_view)
    else:
        al_cs = np.zeros(len(onset_offset))

    return features_clean, al_cs


def detectOnsetOffset(v: np.array, type="saccade") -> np.array:
    '''
    :param v: events vector
    :param type: type of the effents
    :return: onset and offset episodes
    '''

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


def groupingFixation(gaze: np.array, onset_offset:np.array) -> np.array:
    '''
    use this function to group fixation only
    :param gaze: gaze in the world
    :param onset_offset: onset and offset episodes
    :return: the groupped onset and offset episodes
    '''
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


def groupPursuit(onset_offset:np.array) -> np.array:
    '''
    use this function to group smooth pursuit
    :param onset_offset: onset and offset episodes
    :return: the groupped episodes
    '''
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


# methods used previously to detect anticipatory look at phase 1 and 2

# def detectALPhase1(eye_event: np.array, az: np.array, elv: np.array, th=10) -> np.array:
#     '''
#     :param eye_event: eye event of phase 1 from Tobii
#     :param gaze: azimuth (ball-gaze)
#     :param ball: elevation (ball-gaze)
#     :param th: degree/frame (saccade offset - ball trajectory)
#     :return: (angle, onset, offset)
#     '''
#
#     dist_angle = np.sqrt(np.square(az) + np.square(elv))
#     onset_offset = detectOnsetOffset(eye_event == 1)
#
#     al_angle = 1e+4
#     al_offset = 0
#     al_onset = 0
#
#     if len(onset_offset) > 0:
#         for on, off in onset_offset:
#             # angle between ball and gaze after saccade offset
#             min_angle = np.min(dist_angle[off:])
#             if min_angle <= th:
#                 al_offset = off
#                 al_onset = on
#                 al_angle = min_angle
#
#     return np.array([al_angle, al_onset, al_offset])
#
#
# def detectALPhase2(eye_event2: np.array, eye_event3: np.array, gaze: np.array, ball: np.array, th: float = 25,
#                    th_angle_p=25) -> np.array:
#     '''
#     :param eye_event2: eye_event phase 2
#     :param eye_event3: eye_event phase 3
#     :param gaze: gaze in polar coordinates
#     :param ball: ball in polar coordinates
#     :param th: degree/frame (saccade offset - ball trajectory) of AL
#     :param th_angle_p: degree/frame (saccade offset - ball trajectory) of pursuit
#     :return:
#     '''
#
#     # anticipation look
#     al_angle = 1e+4
#     al_magnitude = 1e+4
#     al_offset = 0
#     al_onset = 0
#
#     # pursuit
#     p_avg_angle = 1e+4
#     p_on = 0
#     p_off = 0
#
#     onset_offset_saccade = detectOnsetOffset(eye_event2 == 1)
#
#     ps2_end = len(eye_event2)
#     ps3_start = ps2_end
#     dist_angle = np.sqrt(np.sum(np.square(gaze - ball), -1))  # use azimuth and elevation
#     # dist_angle = np.sqrt(np.square(gaze[:, 0] - ball[:, 0]))  # use azimuth only
#     # detect pursuit
#     dist_angle3 = dist_angle[ps3_start:]
#     pursuit_events_idx = detectOnsetOffset((eye_event3 == 2) & (dist_angle3 <= th_angle_p))
#
#     if (len(onset_offset_saccade) > 0) & (len(pursuit_events_idx) > 0):
#         first_pursuit = pursuit_events_idx[0]
#         last_pursuit = pursuit_events_idx[-1]
#         ball_pursuit = ball[first_pursuit[0], 0]
#         for on, off in onset_offset_saccade:
#             gaze_saccade = gaze[off, 0]
#             dist_sp = np.sqrt(np.sum(np.square(gaze_saccade - ball_pursuit)))
#             # print(dist_sp)
#             if dist_sp <= th:
#                 al_angle = dist_sp
#                 al_onset = on
#                 al_offset = off
#                 al_magnitude = np.sqrt(np.sum(np.square(gaze[off] - gaze[on]))) / ((off - on) + 1e-5)
#
#     if len(pursuit_events_idx) > 0:
#         first_pursuit = pursuit_events_idx[0]
#         last_pursuit = pursuit_events_idx[-1]
#         p_on = first_pursuit[0]
#         p_off = last_pursuit[1]
#         p_avg_angle_list = []
#         for p_on, p_off in pursuit_events_idx:
#             p_avg_angle_list.append(dist_angle3[p_on:p_off])
#
#         p_avg_angle = np.average(np.concatenate(p_avg_angle_list))
#     else:
#         if np.average((eye_event3 == 2) & (dist_angle3 <= th_angle_p)) == 1:
#             p_on = 0
#             p_off = len(dist_angle3) - 1
#             p_avg_angle = np.average(dist_angle3)
#         # else:
#         #     plt.plot(dist_angle3)
#         #     plt.show()
#
#     return np.array([al_angle, al_magnitude, al_onset, al_offset]), np.array([p_avg_angle, p_on, p_off])
