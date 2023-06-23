import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import label



def detectOnsetOffset(v):
    v = v.astype(np.int)
    output = []
    valley_groups, num_groups = label(v == 1)
    if num_groups > 0:
        for i in np.unique(valley_groups)[1:]:
            valley_group = np.argwhere(valley_groups == i)
            output.append([np.min(valley_group) , np.max(valley_group) ])

    return np.asarray(output).astype(int)
def detectALPhase1(eye_event: np.array, az:np.array, elv: np.array, th=10) -> np.array:
    '''
    :param eye_event: eye event of phase 1 from Tobii
    :param az: azimuth (ball-gaze)
    :param elv: elevation (ball-gaze)
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
            if min_angle <=th:
                al_offset = off
                al_onset = on
                al_angle = min_angle

    return np.array([al_angle, al_onset, al_offset])


def detectALPhase2(eye_event2: np.array, eye_event3: np.array, gaze:np.array, ball:np.array,  th:float=25, th_angle_p=15) -> np.array:
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
    al_offset = 0
    al_onset = 0

    # pursuit
    p_avg_angle = 1e+4
    p_on = 0
    p_off = 0

    onset_offset_saccade = detectOnsetOffset(eye_event2 == 1)

    ps2_end = len(eye_event2)
    ps3_start = ps2_end
    dist_angle = np.sqrt(np.sum(np.square(gaze - ball), -1))
    # detect pursuit
    dist_angle3 = dist_angle[ps3_start:]
    pursuit_events_idx = detectOnsetOffset((eye_event3 == 2) & (dist_angle3 <= th_angle_p))

    # plt.plot(dist_angle3)
    # plt.show()
    if (len(onset_offset_saccade) > 0) & (len(pursuit_events_idx) > 0):
        first_pursuit = pursuit_events_idx[0]
        last_pursuit = pursuit_events_idx[-1]
        ball_pursuit = ball[first_pursuit[0], 0]
        for on, off in onset_offset_saccade:
            gaze_saccade = gaze[off, 0]
            dist_sp = np.sqrt(np.sum(np.square(gaze_saccade - ball_pursuit)))
            print(dist_sp)
            if dist_sp <= th:
                al_angle = dist_sp
                al_onset = on
                al_offset = off

        p_on = first_pursuit[0]
        p_off = last_pursuit[1]
        p_avg_angle_list = []
        for p_on, p_off in pursuit_events_idx:
            p_avg_angle_list.append(dist_angle3[p_on:p_off])

        p_avg_angle = np.average(np.concatenate(p_avg_angle_list))


    return np.array([al_angle, al_onset, al_offset]), np.array([p_avg_angle, p_on, p_off])









