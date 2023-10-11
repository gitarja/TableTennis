import numpy as np
from FeaturesEngineering import FeaturesExtractor
class SingleFeaturesExtractor(FeaturesExtractor):

    def extractEpisodeFeatures(self, saccade_normalize=False):

        def episodeGroupLabel():
            if len(self.failures_idx) == 0:
                episode_label = np.expand_dims(np.zeros(len(self.success_idx)), -1)
                observation_label = np.expand_dims(np.arange(0, len(self.success_idx), 1), -1)
                success_label = np.expand_dims(np.ones(len(self.success_idx)), -1)
                sort_idx = np.arange(0, len(self.success_idx), 1)
                return episode_label, observation_label, success_label, sort_idx
            episodes = np.concatenate([self.success_idx[:, 0], self.failures_idx[:, 0]])
            sort_idx = np.argsort(episodes)

            episodes_sorted = episodes[sort_idx]
            failure_start = self.failures_idx[:, 0]
            success_start = self.success_idx[:, 0]
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

        episode_label, observation_label, success_label, sort_idx = episodeGroupLabel()

        # success episodes features
        success_gaze = self.detectGazeEvents(self.success_idx, normalize=saccade_normalize, th_angle_p=25)
        success_fs = self.startForwardSwing(self.success_idx)
        success_start_fs = success_fs["start_fs"][:, 0] if len(self.success_idx) == 1 else success_fs[
            "start_fs"].squeeze()
        success_anglerac = self.angleRacketShoulder(self.success_idx, success_start_fs)
        success_mag_fs = self.speedACCForwardSwing(self.success_idx, success_start_fs)
        success_mag_impact = self.speedACCBeforeImpact(self.success_idx)

        features = np.hstack([success_gaze["gaze_event"],
                              success_gaze["saccade_p1"],
                              success_gaze["saccade_p2"],
                              success_gaze["fsp_phase3"],
                              success_fs["start_fs"],
                              success_anglerac["events_twoseg_angles"],
                              success_mag_fs["racket_speed"],
                              success_mag_fs["racket_acc"],
                              success_mag_fs["ball_speed"],
                              success_mag_fs["ball_acc"],
                              success_mag_fs["racket_speed"] / success_mag_fs["ball_speed"],
                              success_mag_fs["rball_dir"],
                              success_mag_impact["im_racket_force"],
                              success_mag_impact["im_ball_force"],
                              success_mag_impact["im_rb_ratio"],
                              success_mag_impact["im_ball_fimp"],
                              success_mag_impact["im_rb_dist"],
                              success_mag_impact["im_to_wrist_dist"],
                              success_mag_impact["im_rack_wrist_dist"],

                              ])
        # failures episode features
        if len(self.failures_idx):
            failures_gaze = self.detectGazeEvents(self.failures_idx, normalize=saccade_normalize, th_angle_p=25)
            failures_fs = self.startForwardSwing(self.failures_idx)
            failure_start_fs = failures_fs["start_fs"][:, 0] if len(self.failures_idx) == 1 else failures_fs[
                "start_fs"].squeeze()
            failures_anglerac = self.angleRacketShoulder(self.failures_idx, failure_start_fs)
            failures_mag_fs = self.speedACCForwardSwing(self.failures_idx, failure_start_fs)
            failures_mag_impact = self.speedACCBeforeImpact(self.failures_idx)
            failures_features = np.hstack([
                failures_gaze["gaze_event"],
                failures_gaze["saccade_p1"],
                failures_gaze["saccade_p2"],
                failures_gaze["fsp_phase3"],
                failures_fs["start_fs"],
                failures_anglerac["events_twoseg_angles"],
                failures_mag_fs["racket_speed"],
                failures_mag_fs["racket_acc"],
                failures_mag_fs["ball_speed"],
                failures_mag_fs["ball_acc"],
                failures_mag_fs["racket_speed"] / failures_mag_fs["ball_speed"],
                failures_mag_fs["rball_dir"],
                failures_mag_impact["im_racket_force"],
                failures_mag_impact["im_ball_force"],
                failures_mag_impact["im_rb_ratio"],
                failures_mag_impact["im_ball_fimp"],
                failures_mag_impact["im_rb_dist"],
                failures_mag_impact["im_to_wrist_dist"],
                failures_mag_impact["im_rack_wrist_dist"],
            ])

            features = np.vstack([features, failures_features])

        features_sorted = features[sort_idx]

        summary_features = np.hstack([features_sorted, episode_label, observation_label, success_label])

        return summary_features



class DoubleFeaturesExtractor():

    def __init__(self, s1_features:FeaturesExtractor, s2_features:FeaturesExtractor):

        self.s1 = s1_features
        self.s2 = s2_features







