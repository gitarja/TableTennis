import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy import signal
from Utils.Lib import movingAverage
from FeaturesEngineering.FeaturesLib import computeVectorsDirection, computeVelAccV2, computeSegmentAngles, \
    computePhaseCrossCorr, computeNormalizedCrossCorr, computeSpectralCoherence, computeMutualInf, computeTransferEntropy, computeNormalizedED
from Utils.Lib import cartesianToSpher, computePowerSpectrum
from Utils.Interpolate import interpolate
from scipy.ndimage import label
from SynchAnalysis.Preprocessing.Filters import butterBandpassFilter
from SynchAnalysis.SynchMetrices import computePhaseSync, computeEntropySynchIndex, computeFourierModeSynchIndex
from SynchAnalysis.SynchTesting import generateIAAFTData, generateFourierPhaseSurrogate
from scipy.signal import welch, hilbert


def detectGazeEvent( gaze: np.array, ball: np.array, tobii: np.array):
    gaze_avg = np.array([movingAverage(gaze[:, i], n=2) for i in range(3)]).transpose()

    ball = np.array([interpolate(ball[:, i]) for i in range(3)]).transpose()
    ball_avg = np.array([movingAverage(ball[:, i], n=2) for i in range(3)]).transpose()

    tobii = np.array([movingAverage(tobii[:, i], n=1) for i in range(3)]).transpose()
    tobii_avg = np.array([movingAverage(tobii[:, i], n=3) for i in range(3)]).transpose()
    onset_offset_saccade, onset_offset_sp, onset_offset_fix, stream_label = detectSaccade(gaze_avg, ball_avg, tobii,
                                                                                          tobii_avg)

    return onset_offset_saccade, onset_offset_sp, onset_offset_fix, stream_label

def detectSaccade(gaze: np.array, ball: np.array, tobii:np.array,  tobii_avg:np.array, th_pursuit:float=10):
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

    # relabel saccade
    for on, off in onset_offset_saccade:
        labels_saccade[on:off+1] = 1

    # detect pursuit
    # ball_gaze_dist = computeSegmentAngles(ball - tobii, gaze - tobii)
    labels_sp = np.zeros(N_label)
    vel_ratio =   vel_gaze_h / (vel_ball + 1e-15)
    _, gaze_az, gaze_elv = cartesianToSpher(vector=gaze - tobii_avg, swap=False)
    _, ball_az, ball_elv = cartesianToSpher(vector=ball - tobii_avg, swap=False)

    # when computing the angle between the gaze and the ball, both of them shuld be converted into visual field
    gaze_view = np.vstack([gaze_az, gaze_elv]).transpose()
    ball_view = np.vstack([ball_az, ball_elv]).transpose()


    ball_gaze_dist = np.linalg.norm(gaze_view - ball_view, axis=-1)
    # labels_sp[(ball_gaze_dist<=15) & ((vel_gaze_h >30) & (vel_ratio >= 0.3) & (vel_ratio <= 2.0))] = 1  # pursuit gain
    # setting the dist to 5 will reduce the number of smooth pursuit, better to use 10
    labels_sp[((ball_gaze_dist<=th_pursuit) | ((vel_gaze_h >30) & ((vel_ratio >= 0.3) & (vel_ratio <= 1.2))))] = 1  # pursuit gain
    labels_sp[labels_saccade==1] = 0
    onset_offset_sp = detectOnsetOffset(labels_sp == 1, type="sp")
    onset_offset_sp = groupPursuit(onset_offset_sp)

    # if (N_label) > 90:
    #     print(N_label)
    #     plt.close()
    #     plt.subplot(3, 1, 1)
    #     plt.plot(gaze_view[:, 0], color="blue")
    #     plt.plot(ball_view[:, 0], color="red")
    #     plt.subplot(3, 1, 2)
    #     plt.plot(gaze_view[:, 1], color="blue")
    #     plt.plot(ball_view[:, 1], color="red")
    #     plt.subplot(3, 1, 3)
    #     plt.plot(ball_gaze_dist)
    #     plt.savefig("F:\\users\\prasetia\\projects\\Animations\\TableTennis\\ball_gaze_dist.eps", format='eps')
    #     plt.show()

    # detect

    labels_fix = np.ones(N_label)
    for on, off in onset_offset_saccade:
        labels_fix[on:off+1] = 0
    for on, off in onset_offset_sp:
        labels_fix[on:off + 1] = 0
    onset_offset_fix = detectOnsetOffset(labels_fix == 1, type="fix")
    onset_offset_fix = groupingFixation(gaze, onset_offset_fix)

    stream_label = contstructStreamLabel(onset_offset_fix, onset_offset_sp, onset_offset_saccade)

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

def saccadeFeatures(onset_offset: np.array, gaze: np.array, ball: np.array, win_length:int = 5, phase_start=0, phase_end=100, classify_al=True, phase_2=False) -> tuple:
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
    Peak velocity               =pv
    Direction ball              =db direction relative to the ball trajectory
    Saccade Deceleration        =bgd how the saccade decelerates
    Saccade duration ration     =sdr saccade_duration / phase_duration



    :param onset_offset: onset and offset episodes
    :param gaze: gaze in the world (gaze-tobii)
    :param ball: ball in the world  (ball-tobii)
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




    features = []

    if len(onset_offset) > 0:
        for on, off in onset_offset:
            # print(on)
            # extract saccade features

            if ((off - on) > 0):
                ball_view_s = ball_view[on:off + 1]


                gm = computeSegmentAngles(np.expand_dims(gaze[on], 0), np.expand_dims(gaze[off], 0))[0]



                gaze_at_offset = gaze_view[off]
                ball_at_bounce = ball_view[phase_end - 1]  # ball position at the bounce event
                after_ofset_angle = np.linalg.norm(gaze_at_offset - ball_at_bounce, axis=-1)

                if np.sum(np.isnan(ball_view_s)):
                    print("error")
                # sdr = ((off - on)  / (phase_end - phase_start)) * 10


                on_event = (on - phase_start) * 10
                off_event = (off - phase_start) * 10
                # concatenate saccades features
                try:
                    if after_ofset_angle < 150:
                        features.append([
                                         on_event,
                                         after_ofset_angle,
                                         gm,
                                         ])
                    else:
                        features.append([
                            np.nan,
                            np.nan,
                            np.nan,
                        ])

                except:
                    print("error")

    # remove nan features

    features = np.asarray(features)
    features[np.isnan(features)] = 0

    # classify whether a saccade is anticipitary look or correction saccade
    if classify_al:
        al_cs = classifyALCS(onset_offset, gaze, gaze_view, ball_view)
    else:
        al_cs = np.zeros(len(onset_offset))

    return features, al_cs


def fixationSPFeatures(onset_offset_fsp:np.array, onset_offset_al_p2:np.array, gaze: np.array, ball: np.array,  phase_start=0, phase_end=100, phase2_start=0) -> np.array:
    '''
    We extract fixation / smooth pursuit features that occurs after AL in phase 2.
    The tracking accuracy computes the accuracy of fixation or smooth pursuit from when it starts until ball interception.

    Onset is the start of the smooth pursuit. If there are two separated smooth pursuit, we only consider the first one in terms of onset and duration

    Extract features of fsp that occurs immediately after AL in the P2
    :param onset_offset_fsp: a list containing onset and offset of fx / sp
    :param onset_offset_al_p2: a list containing the onset and offset of AL in phase 2
    :param gaze: gaze vector in global coordinate
    :param ball: ball vector in global coordinate
    :return:
    onset of the first fx/sp
    duration of the first fx/sp
    tracking accuracy
    '''

    idx = onset_offset_fsp[:, 0] > onset_offset_al_p2[0,1]
    # print(idx)
    if np.sum(idx) > 0:
        features = []
        on, off = onset_offset_fsp[np.argwhere(idx)[0,0]]

        if (off - on) > 2:
            _, gaze_az, gaze_elv = cartesianToSpher(vector=gaze, swap=False)
            _, ball_az, ball_elv = cartesianToSpher(vector=ball, swap=False)

            # when computing the angle between the gaze and the ball, both of them should be converted into visual field
            gaze_view = np.vstack([gaze_az, gaze_elv]).transpose()
            ball_view = np.vstack([ball_az, ball_elv]).transpose()
            dist_angle = np.linalg.norm(gaze_view - ball_view, axis=-1)


            # compute the features
            on_event = (on - phase2_start)  * 10

            # mean of pursuit
            mean_dist = np.nanmean(dist_angle[on:off])


            phase_dist = np.sqrt(np.average(np.square(dist_angle[on:] - mean_dist)))
            # tracking_accuracy = np.sqrt(np.average(np.square(gaze_view[on:] - ball_view[on:])))

            features.append([
                on_event,
                (off - on) * 10,
                phase_dist,


            ])

            features = np.asarray(features)

            return features

    sp_features = np.empty((1, 3))
    sp_features[:] = np.nan
    return sp_features





def episodeFeatures(v: np.array, min_episode:float = 1) -> np.array:
    # initialize output
    # v = np.argwhere(v == 1)
    output = []

    if len(v) > 2:

        # label groups of sample that belong to the same peak
        valley_groups, num_groups = label(v)

        for i in np.unique(valley_groups)[1:]:
            point_group = np.argwhere(valley_groups == i)
            if np.max(point_group) - np.min(point_group) > min_episode:
                output.append([np.min(point_group), np.max(point_group)])

    episodes =  np.array(output)
    if len(episodes) > 0:
        duration = episodes[:, 1] - episodes[:, 0]

        avg_duration = np.average(duration)
        min_duration = np.min(duration)
        max_duration = np.max(duration)
        percentage = np.average(v)
        return min_duration, max_duration, avg_duration, percentage
    else:

        return 0, 0, 0, 0


def convertToVisualFieldView(v) -> np.array:

    _, az, elv = cartesianToSpher(vector=v, swap=False)
    field_view = np.vstack([az, elv]).transpose()

    return field_view


def jointAttentionFeatures(g1, g2, ball1, ball2, th=10):
    '''
    :param g1: gaze view of subject 1
    :param g2: gaze view of subject 2
    :param ball1: ball view of subject 1
    :param ball2: ball view of subject 2
    :param th: threshold for joint attention
    :return: features related to the duration of joint attention
    '''

    # gaze and ball view of subject 1
    gaze_view_s1 = convertToVisualFieldView(g1)
    ball_view_s1 = convertToVisualFieldView(ball1)


    # gaze and ball view of subject 2
    gaze_view_s2 = convertToVisualFieldView(g2)
    ball_view_s2 = convertToVisualFieldView(ball2)

    dist1 = movingAverage(np.linalg.norm(gaze_view_s1 - ball_view_s1, axis=-1), n=3)

    dist2 = movingAverage(np.linalg.norm(gaze_view_s2 - ball_view_s2, axis=-1), n=3)


    mask = (dist1 <= th) & (dist2 <= th)

    joint_attention_features = episodeFeatures(mask, min_episode=5)[-1]




    return joint_attention_features



def gazeMovemenCoorientationtFeatures(g1:np.array, g2:np.array, ball1:np.array, ball2:np.array, full=False):
    '''
    :param g1: gaze vector (gaze in the head) of the first individual
    :param g2: gaze vector of the second individual
    :param ball1: ball in the head of the first individual
    :param ball2: ball in the head of the second individual
    :return:
    '''
    # print(len(g1))
    # gaze and ball view of subject 1
    gaze_view_s1 = convertToVisualFieldView(g1)
    ball_view_s1 = convertToVisualFieldView(ball1)


    # gaze and ball view of subject 2
    gaze_view_s2 = convertToVisualFieldView(g2)
    ball_view_s2 = convertToVisualFieldView(ball2)

    dist1 = np.linalg.norm(gaze_view_s1 - ball_view_s1, axis=-1)
    filtered_dist1 = np.expand_dims(butterBandpassFilter(dist1, 1, 5, 100), 0)
    _, _, vel_gaze_h1 = computeVelAccV2(g1, normalize=True)


    dist2 = np.linalg.norm(gaze_view_s2 - ball_view_s2, axis=-1)
    filtered_dist2 = np.expand_dims(butterBandpassFilter(dist2, 1, 5, 100), 0)
    _, _, vel_gaze_h2 = computeVelAccV2(g2, normalize=True)



    sync_index_gaze = computeFourierModeSynchIndex(filtered_dist1, filtered_dist2, n=1, m=1)
    correlation_gaze_velocity = computeNormalizedCrossCorr(vel_gaze_h1, vel_gaze_h2, th=5, normalize=True)


    return np.hstack([correlation_gaze_velocity, sync_index_gaze])


# def gazeMovemenCoorientationtFeatures(g1:np.array, g2:np.array, ball1:np.array, ball2:np.array, full=False):
#     '''
#     :param g1: gaze vector (gaze in the head) of the first individual
#     :param g2: gaze vector of the second individual
#     :param ball1: ball in the head of the first individual
#     :param ball2: ball in the head of the second individual
#     :return:
#     '''
#     # print(len(g1))
#     # gaze and ball view of subject 1
#     gaze_view_s1 = convertToVisualFieldView(g1)
#     ball_view_s1 = convertToVisualFieldView(ball1)
#
#
#     # gaze and ball view of subject 2
#     gaze_view_s2 = convertToVisualFieldView(g2)
#     ball_view_s2 = convertToVisualFieldView(ball2)
#
#     dist1 = np.linalg.norm(gaze_view_s1 - ball_view_s1, axis=-1)
#     filtered_dist1 = np.expand_dims(butterBandpassFilter(dist1, 1, 5, 100), 0)
#     _, vel_gaze_h1, _ = computeVelAccV2(g1, normalize=False)
#
#
#     dist2 = np.linalg.norm(gaze_view_s2 - ball_view_s2, axis=-1)
#     filtered_dist2 = np.expand_dims(butterBandpassFilter(dist2, 1, 5, 100), 0)
#     _, vel_gaze_h2, _ = computeVelAccV2(g2, normalize=False)
#
#
#     # avg_phase_diff, dist1_phase, dist2_phase = computePhaseSync(filtered_dist1, filtered_dist2, nt=50, n=1, m=1, return_phase=True, normalized=False)
#     # sync_index = computeEntropySynchIndex(filtered_dist1, filtered_dist2, n_bins=7, n=1, m=1)
#     sync_index = computeFourierModeSynchIndex(filtered_dist1, filtered_dist2, n=1, m=1)
#
#     # surrogate
#     # surr1_avg_phase_diff, dist1_phase, dist2_phase = computePhaseSync(filtered_dist1, generateFourierPhaseSurrogate(filtered_dist2), nt=50, n=1, m=1,
#     #                                                             return_phase=True, normalized=False)
#     surr1_sync_index = computeFourierModeSynchIndex(filtered_dist1, generateFourierPhaseSurrogate(filtered_dist2, gaussian=True), n=1, m=1)
#
#
#
#
#     # print("Phase sync %f , surrogate %f" % (avg_phase_diff, surr1_avg_phase_diff))
#     # import matplotlib.pyplot as plt
#
#     # step 1
#     # fig, axs = plt.subplots(3)
#     # noise = np.random.permutation(dist1)
#     # f1, p1 = computePowerSpectrum(dist1)
#     # nf1, np1 = computePowerSpectrum(noise)
#     #
#     # print(np.sqrt(np.average(np.square(p1 - np1))))
#     # axs[0].plot(dist1, color='#525252')
#     # axs[1].plot(noise, color='#f03b20')
#     #
#     # axs[2].plot(f1, p1, color='#525252')
#     # axs[2].plot(nf1, np1, color='#f03b20')
#     # plt.show()
#
#
#     # plot sync with surrogate
#     # fig, axs = plt.subplots(2)
#     # axs[0].plot(avg_phase_diff, color='#252525')
#     # axs[0].plot(surr1_avg_phase_diff, color='#636363')
#     #
#     # axs[1].plot(sync_index, color='#252525')
#     # axs[1].plot(surr1_sync_index, color='#636363')
#
#
#     # plt.plot(filtered_dist2[0])
#     # plt.plot(generateIAAFTData(filtered_dist2, nmax_iter=7)[0])
#     # plt.show()
#
#     # plot hist phase difference
#
#     # plt.hist(dist1_phase.flatten() - dist2_phase.flatten(), bins=100)
#     # plt.show()
#
#     # Step 2: plot spectrum and effect of filtering
#     # f1, p1 = computePowerSpectrum(dist1)
#     # filtered_f1, filtered_p1 = computePowerSpectrum(filtered_dist1[0])
#     # fig, axs = plt.subplots(4)
#     # axs[0].plot(dist1, color='#525252')
#     # axs[1].plot(filtered_dist1[0], color='#f03b20')
#     # axs[2].plot(vel_gaze_h1, color='#525252')
#     #
#     # axs[3].plot(f1, p1, color='#525252')
#     # axs[3].plot(filtered_f1, filtered_p1, color='#f03b20')
#     # plt.show()
#
#
#     # plot sync results
#     # fig, axs = plt.subplots(3)
#     # axs[0].plot(np.angle(hilbert(filtered_dist1[0])), color='#cb181d')
#     # axs[0].plot(np.angle(hilbert(filtered_dist2[0])), color='#2171b5')
#     #
#     # axs[1].plot(avg_phase_diff, color='#525252')
#     # axs[2].plot(sync_index, color='#525252')
#     # plt.show()
#
#
#
#     # plt.show()
#
#     return sync_index, surr1_sync_index

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
                th_duration = 0
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
        # clean sp less than 50 ms
        for i in range(len(new_onset_offset)):
            on, off = new_onset_offset[i]
            if (off - on) < 5:
                new_onset_offset[i] = new_onset_offset[i] * 0

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
