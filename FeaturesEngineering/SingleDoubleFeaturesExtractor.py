import numpy as np
from FeaturesEngineering.FeaturesExtractor import FeaturesExtractor
from FeaturesEngineering.SubjectObject import Subject, Ball, TableWall
from Utils.Lib import percentageChange, relativeDifference


def episodeGroupLabel(success_idx, failures_idx):
    if len(failures_idx) == 0:
        episode_label = np.expand_dims(np.zeros(len(success_idx)), -1)
        observation_label = np.expand_dims(np.arange(0, len(success_idx), 1), -1)
        success_label = np.expand_dims(np.ones(len(success_idx)), -1)
        sort_idx = np.arange(0, len(success_idx), 1)
        return episode_label, observation_label, success_label, sort_idx
    episodes = np.concatenate([success_idx[:, 0], failures_idx[:, 0]])
    sort_idx = np.argsort(episodes)

    episodes_sorted = episodes[sort_idx]
    failure_start = failures_idx[:, 0]
    success_start = success_idx[:, 0]
    episode_group_idx = 0
    observation_idx = 0
    episode_label_list = []
    success_label_list = []
    observation_label_list = []

    for s in episodes_sorted:

        episode_label_list.append([episode_group_idx])
        observation_label_list.append([observation_idx])
        if s in failure_start:
            episode_group_idx += 1
            if np.sum(success_start > s) == 0:
                success_label_list.append([-1])
            else:
                success_label_list.append([0])
        else:
            success_label_list.append([1])
        observation_idx += 1

    return np.asarray(episode_label_list), np.asarray(observation_label_list), np.asarray(
        success_label_list), sort_idx


class SingleFeaturesExtractor(FeaturesExtractor):

    def __init__(self, sub: Subject, ball: Ball, table_wall: TableWall):
        super().__init__(sub, ball, table_wall)

    def extractEpisodeFeatures(self, saccade_normalize=False):
        episode_label, observation_label, success_label, sort_idx = episodeGroupLabel(self.b.success_idx,
                                                                                      self.b.failures_idx)

        # success episodes features
        success_gaze = self.detectGazeEvents(self.b.success_idx, normalize=saccade_normalize, th_angle_p=25)
        success_fs = self.startForwardSwing(self.b.success_idx)
        success_start_fs = success_fs["start_fs"][:, 0] if len(self.b.success_idx) == 1 else success_fs[
            "start_fs"].squeeze()
        success_bouncing_point = self.bouncingPoint(self.b.all_episodes, self.b.success_idx, centro=self.tw.wall_top)
        success_mag_fs = self.speedACCForwardSwing(self.b.success_idx, success_start_fs)
        success_mag_impact = self.speedACCBeforeImpact(self.b.success_idx)

        features = np.hstack([success_gaze["gaze_event"],
                              success_gaze["saccade_p1"],
                              success_gaze["saccade_p2"],
                              success_gaze["fsp_phase3"],
                              success_fs["start_fs"] * 10,
                              success_mag_fs["ball_speed"] / success_mag_fs["racket_speed"],
                              success_mag_fs["rball_dir"],
                              success_mag_fs["rball_dist"],
                              success_mag_impact["im_racket_force"],
                              success_mag_impact["im_ball_force"],
                              success_mag_impact["im_rb_ang_collision"],
                              success_mag_impact["im_rb_dist"],
                              success_mag_impact["im_rack_wrist_dist"],
                              success_bouncing_point["bouncing_position"]

                              ])
        # failures episode features
        if len(self.b.failures_idx):
            failures_gaze = self.detectGazeEvents(self.b.failures_idx, normalize=saccade_normalize, th_angle_p=25)
            failures_fs = self.startForwardSwing(self.b.failures_idx)
            failure_start_fs = failures_fs["start_fs"][:, 0] if len(self.b.failures_idx) == 1 else failures_fs[
                "start_fs"].squeeze()
            failures_bouncing_point = self.bouncingPoint(self.b.all_episodes, self.b.failures_idx, centro=self.tw.wall_top)
            failures_mag_fs = self.speedACCForwardSwing(self.b.failures_idx, failure_start_fs)
            failures_mag_impact = self.speedACCBeforeImpact(self.b.failures_idx)
            failures_features = np.hstack([
                failures_gaze["gaze_event"],
                failures_gaze["saccade_p1"],
                failures_gaze["saccade_p2"],
                failures_gaze["fsp_phase3"],
                failures_fs["start_fs"] * 10,
                failures_mag_fs["racket_speed"] / failures_mag_fs["ball_speed"],
                failures_mag_fs["rball_dir"],
                failures_mag_fs["rball_dist"],
                failures_mag_impact["im_racket_force"],
                failures_mag_impact["im_ball_force"],
                failures_mag_impact["im_rb_ang_collision"],
                failures_mag_impact["im_rb_dist"],
                failures_mag_impact["im_rack_wrist_dist"],
                failures_bouncing_point["bouncing_position"]
            ])

            features = np.vstack([features, failures_features])

        features_sorted = features[sort_idx]

        summary_features = np.hstack([features_sorted, episode_label, observation_label, success_label])

        return summary_features


class DoubleFeaturesExtractor():
    '''
    not for episodes
    e[0] = start of episode
    e[1] = end of episode
    e[2] = bounce 1
    e[3] = bounce 2
    e[4] = which racket hit the ball (1: first racket, 2: second racket)
    e[5] = which racket does not hit the ball (1: first racket, 2: second racket)
    e[6] = who is the receiver (0: first subject, 1: second subject)
    e[7] = who is the hitter (0: first subject, 1: second subject)
    e[8] = the pair of the episode
    '''

    def __init__(self, s1_features: FeaturesExtractor, s2_features: FeaturesExtractor, ball: Ball,
                 table_wall: TableWall):

        self.s1 = s1_features
        self.s2 = s2_features
        self.b = ball
        self.table_wall = table_wall

        success_hitter_receiver, failure_hitter_receiver = self.identifyHitterReceiver()
        success_pair, failure_pair = self.identifyPairEpisodes()

        self.b.success_idx = np.hstack([self.b.success_idx, success_hitter_receiver, success_pair])

        if len(failure_hitter_receiver) > 0:
            self.b.failures_idx = np.hstack([self.b.failures_idx, failure_hitter_receiver, failure_pair])

    def identifyHitterReceiver(self):
        '''
        Hitter: the person that hit the ball in Phase 1
        Receiver: the person that hit the ball in Phase 3
        :return:
        '''

        # success episodes
        hit_ball_impact1 = self.b.ball_t[self.b.success_idx[:, 0]]
        hit_ball_impact2 = self.b.ball_t[self.b.success_idx[:, 1]]

        # first impact
        r11 = self.s1.p.racket_segment_T[self.b.success_idx[:, 0]]
        r12 = self.s2.p.racket_segment_T[self.b.success_idx[:, 0]]

        # second impact
        r1 = self.s1.p.racket_segment_T[self.b.success_idx[:, 1]]
        r2 = self.s2.p.racket_segment_T[self.b.success_idx[:, 1]]


        receiver_success_idx = np.argmin(np.array(
            [np.linalg.norm(hit_ball_impact2 - r1, axis=-1), np.linalg.norm(hit_ball_impact2 - r2, axis=-1)]).T,
                                       axis=-1)
        hitter_success_idx = np.argmin(
            np.array(
                [np.linalg.norm(hit_ball_impact1 - r11, axis=-1), np.linalg.norm(hit_ball_impact1 - r12, axis=-1)]).T,
            axis=-1)

        # failures episodes
        if len(self.b.failures_idx) > 0:
            hit_ball_impact1 = self.b.ball_t[self.b.failures_idx[:, 0]]
            hit_ball_impact2 = self.b.ball_t[self.b.failures_idx[:, 1]]
            r1 = self.s1.p.racket_segment_T[self.b.failures_idx[:, 1]]
            r2 = self.s2.p.racket_segment_T[self.b.failures_idx[:, 1]]

            r11 = self.s1.p.racket_segment_T[self.b.failures_idx[:, 0]]
            r12 = self.s2.p.racket_segment_T[self.b.failures_idx[:, 0]]
            receiver_failure_idx = np.argmin(
                np.array(
                    [np.linalg.norm(hit_ball_impact2 - r1, axis=-1), np.linalg.norm(hit_ball_impact2 - r2, axis=-1)]).T,
                axis=-1)
            hitter_failure_idx = np.argmin(
                np.array([np.linalg.norm(hit_ball_impact1 - r11, axis=-1),
                          np.linalg.norm(hit_ball_impact1 - r12, axis=-1)]).T, axis=-1)

            return np.array([receiver_success_idx, hitter_success_idx]).T, np.array(
                [receiver_failure_idx, hitter_failure_idx]).T

        return np.array([receiver_success_idx, hitter_success_idx]).T, np.array([])

    def identifyPairEpisodes(self):

        success_pair_episode = []
        for e in self.b.success_idx:
            prev_ep_idx = np.argwhere(e[0] == self.b.success_idx[:, 1])
            if len(prev_ep_idx) == 0:
                success_pair_episode.append(-1)
            else:
                success_pair_episode.append(prev_ep_idx[0, 0])

        failure_pair_episode = []
        for e in self.b.failures_idx:
            prev_ep_idx = np.argwhere(e[0] == self.b.success_idx[:, 1])
            if len(prev_ep_idx) == 0:
                failure_pair_episode.append(-1)
            else:
                failure_pair_episode.append(prev_ep_idx[0, 0])

        return np.expand_dims(success_pair_episode, -1), np.expand_dims(failure_pair_episode, -1)

    # def extractPairEpisodeFeatures(self):
    #     episode_label, observation_label, success_label, sort_idx = episodeGroupLabel(self.b.success_idx,
    #                                                                                   self.b.failures_idx)
    #
    #     sucess_hitting_sim = self.extractPairFeatures(self.s1, self.s2, self.b.success_idx, self.b.success_idx)
    #     features = sucess_hitting_sim
    #
    #     if len(self.b.failures_idx):
    #         failures_hitting_sim = self.extractPairFeatures(self.s1, self.s2, self.b.success_idx, self.b.failures_idx)
    #
    #         features = np.vstack([features, failures_hitting_sim])
    #
    #     features_sorted = features[sort_idx]
    #
    #     return features_sorted

    def extractPairFeatures(self, s1, s2, ref_episodes, episodes, ref_features_imp, features_imp):
        subjects = [s1, s2]
        features = np.empty((len(episodes), 3 + ref_features_imp.shape[1]))
        features[:] = np.nan
        ball = self.b.ball_t
        j = 0
        for i in range(len(episodes)):
            if episodes[i][-1] != -1:
                prev_e = ref_episodes[episodes[i][-1]]
                current_e = episodes[i]

                hitter_1 = subjects[prev_e[6]]
                hitter_2 = subjects[current_e[6]]

                prev_features = ref_features_imp[episodes[i][-1]]
                current_features = features_imp[i]

                ball_bounce_prev = ball[prev_e[2]]
                ball_bounce_cur = ball[current_e[2]]

                dist_bounce = np.linalg.norm(ball_bounce_prev - ball_bounce_cur)

                features_diff = np.abs(prev_features - current_features)

                speed_racket_s1_vel = hitter_1.extractSpeedForwardSwing(prev_e[2], prev_e[1])

                racket_speed_sim = hitter_2.racketMovementSimilarity(current_e[2], current_e[1], speed_racket_s1_vel)

                features[j] = np.hstack(
                    [racket_speed_sim["dtw_dist"], racket_speed_sim["lcc_dist"], dist_bounce, features_diff])

                # speed_racket_s2_vel =  hitter_2.extractSpeedForwardSwing(current_e[2], current_e[1])
                # import matplotlib.pyplot as plt
                #
                # plt.plot(speed_racket_s1_vel, label="prev")
                # plt.plot(speed_racket_s2_vel, label="current")
                # plt.legend()
                # print(racket_speed_sim)
                # plt.show()

            j += 1

        return features

    def extractEpisodeFeatures(self, saccade_normalize=False):

        episode_label, observation_label, success_label, sort_idx = episodeGroupLabel(self.b.success_idx,
                                                                                      self.b.failures_idx)
        # success gaze events
        s1_success_gaze, s1_success_p1on, s1_success_p2on = self.extractGazeFeatures(self.s1, self.b.success_idx,
                                                                                     saccade_normalize, iam_idx=0)
        s2_success_gaze, s2_success_p1on, s2_success_p2on = self.extractGazeFeatures(self.s2, self.b.success_idx,
                                                                                     saccade_normalize, iam_idx=1)

        receiver_success_gaze, hitter_success_gaze = self.selectReceiverFeatures(self.b.success_idx, np.array(
            [s1_success_gaze, s2_success_gaze]))

        # success ball position to root

        # success forward swing and impact
        s1_success_fs_impact = self.extractForwardSwingImpact(self.s1, self.b.success_idx, self.s2)
        s2_success_fs_impact = self.extractForwardSwingImpact(self.s2, self.b.success_idx, self.s1)
        receiver_success_fs_impact, _ = self.selectReceiverFeatures(self.b.success_idx,
                                                                np.array([s1_success_fs_impact, s2_success_fs_impact]))

        #  spatial hit
        spatial_hit_success = self.s1.distanceDoubleHit(self.b.success_idx, self.s2)
        receiver_success_fs_impact = np.hstack([receiver_success_fs_impact, spatial_hit_success])
        # joint attention and gaze sync
        joint_attention_success = self.s1.detectJointAttention(self.b.success_idx, self.s2, b=self.b)

        features = np.hstack([receiver_success_gaze,
                              hitter_success_gaze,
                              joint_attention_success["joint_attention"],
                              receiver_success_fs_impact,
                              self.b.success_idx[:, 6:]

                              ])

        if len(self.b.failures_idx):
            # failures gaze events
            s1_failure_gaze, s1_failures_p1on, s1_failures_p2on = self.extractGazeFeatures(self.s1, self.b.failures_idx,
                                                                                           saccade_normalize, iam_idx=0)
            s2_failure_gaze, s2_failures_p1on, s2_failures_p2on = self.extractGazeFeatures(self.s2, self.b.failures_idx,
                                                                                           saccade_normalize, iam_idx=1)



            receiver_failure_gaze, hitter_failure_gaze = self.selectReceiverFeatures(self.b.failures_idx, np.array(
                [s1_failure_gaze, s2_failure_gaze]))

            # failures forward swing and impact
            s1_failures_fs_impact = self.extractForwardSwingImpact(self.s1, self.b.failures_idx, self.s2)
            s2_failures_fs_impact = self.extractForwardSwingImpact(self.s2, self.b.failures_idx, self.s1)

            receiver_failures_fs_impact, _ = self.selectReceiverFeatures(self.b.failures_idx,
                                                                     np.array([s1_failures_fs_impact,
                                                                               s2_failures_fs_impact]))

            #  spatial hit
            spatial_hit_failure = self.s1.distanceDoubleHit(self.b.failures_idx, self.s2)
            receiver_failures_fs_impact = np.hstack([receiver_failures_fs_impact, spatial_hit_failure])
            # joint attention and gaze sync
            joint_attention_failure = self.s1.detectJointAttention(self.b.failures_idx, self.s2, b=self.b)
            failures_features = np.hstack([
                receiver_failure_gaze,
                hitter_failure_gaze,
                joint_attention_failure["joint_attention"],
                receiver_failures_fs_impact,
                self.b.failures_idx[:, 6:]

            ])

            features = np.vstack([features, failures_features])

        features_sorted = features[sort_idx]

        summary_features = np.hstack([features_sorted, episode_label, observation_label, success_label])
        return summary_features

    def extractGazeFeatures(self, s, episodes, saccade_normalize=False, iam_idx=0):
        '''
        :param s:
        :param episodes:
        :param saccade_normalize:
        :param iam_idx: i am the first (0) or the second (1) subject
        :return:
        '''
        gaze_features = s.detectGazeEvents(episodes, normalize=saccade_normalize, th_angle_p=25,
                                           fill_in=False, iam_idx=iam_idx, double=True)

        features = np.hstack([gaze_features["gaze_event"],
                              gaze_features["saccade_p1"],
                              gaze_features["saccade_p2"],
                              gaze_features["fsp_phase3"]])

        return features, np.expand_dims(gaze_features["saccade_p1"][:, 0], -1), np.expand_dims(
            gaze_features["saccade_p2"][:, 0], -1)

    def extractForwardSwingImpact(self, s, episodes, ref):
        # extract subject 1 features
        fs = s.startForwardSwing(episodes)
        start_fs = fs["start_fs"][:, 0] if len(episodes) == 1 else fs[
            "start_fs"].squeeze()
        mag_fs = s.speedACCForwardSwing(episodes, start_fs, n_window=5)
        mag_impact = s.speedACCBeforeImpact(episodes, win_length=3)

        # bouncing
        bouncing_points = s.bouncingPoint(self.b.all_episodes, episodes, ref=ref.p.lower_back_segment_T)

        # impact
        impact_points = s.impactPoint(episodes, ref=ref)

        features = np.hstack([
            fs["start_fs"] * 10,
            mag_fs["ball_speed"] / mag_fs["racket_speed"],
            mag_fs["rball_dir"],
            mag_fs["rball_dist"],
            mag_impact["im_racket_force"],
            mag_impact["im_ball_force"],
            mag_impact["im_rb_ang_collision"],
            mag_impact["im_rb_dist"],
            mag_impact["im_rack_wrist_dist"],
            bouncing_points["bouncing_position"],
            impact_points["impact_position"]

        ])
        return features

    def computePairedFeatures(self, episodes, fill_in=False):

        # extract subject 1 features
        s1_fs = self.s1.startForwardSwing(episodes)
        s1_start_fs = s1_fs["start_fs"][:, 0] if len(episodes) == 1 else s1_fs[
            "start_fs"].squeeze()
        s1_mag_fs = self.s1.speedACCForwardSwing(episodes, s1_start_fs)
        s1_failures_mag_impact = self.s1.speedACCBeforeImpact(episodes)

        # extract subject 1 features
        s2_fs = self.s2.startForwardSwing(episodes)
        s2_start_fs = s2_fs["start_fs"][:, 0] if len(episodes) == 1 else s2_fs[
            "start_fs"].squeeze()
        s2_mag_fs = self.s2.speedACCForwardSwing(episodes, s2_start_fs)
        s2_failures_mag_impact = self.s2.speedACCBeforeImpact(episodes)

        # construct subject 1 features
        s1_features = np.hstack([
            s1_fs["start_fs"] * 10,
            s1_mag_fs["racket_speed"],
            s1_mag_fs["racket_speed"] / s1_mag_fs["ball_speed"],
            s1_mag_fs["rball_dir"],
            s1_mag_fs["rball_dist"],
            s1_failures_mag_impact["im_racket_force"],
            s1_failures_mag_impact["im_rb_ang_collision"],
            s1_failures_mag_impact["im_rb_dist"],
            s1_failures_mag_impact["im_rack_wrist_dist"],

        ])

        # construct subject 2 features
        s2_features = np.hstack([
            s2_fs["start_fs"] * 10,
            s2_mag_fs["racket_speed"],
            s2_mag_fs["racket_speed"] / s2_mag_fs["ball_speed"],
            s2_mag_fs["rball_dir"],
            s2_mag_fs["rball_dist"],
            s2_failures_mag_impact["im_racket_force"],
            s2_failures_mag_impact["im_rb_ang_collision"],
            s2_failures_mag_impact["im_rb_dist"],
            s2_failures_mag_impact["im_rack_wrist_dist"],

        ])
        if fill_in:
            s1_features[np.isnan(s1_features)] = 0
            s2_features[np.isnan(s2_features)] = 0
        # s1_features[episodes[:, 6] == 1] = np.nan
        # s2_features[episodes[:, 6] == 0] = np.nan
        features = np.array([s1_features, s2_features])
        return features

    def selectReceiverFeatures(self, episodes, features):
        receiver_features = []
        hitter_features = []
        i = 0
        for e in episodes:
            receiver_features.append(features[e[6], i])
            hitter_features.append(features[e[7], i])
            i += 1

        return np.asarray(receiver_features), np.asarray(hitter_features)
