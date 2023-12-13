import numpy as np
from FeaturesEngineering.FeaturesExtractor import  FeaturesExtractor
from SubjectObject import Subject, Ball, TableWall

class SingleFeaturesExtractor(FeaturesExtractor):

    def __init__(self, sub: Subject, ball: Ball, table_wall: TableWall):
        super().__init__(sub, ball, table_wall)



    def extractEpisodeFeatures(self, saccade_normalize=False):

        def episodeGroupLabel():
            if len(self.b.failures_idx) == 0:
                episode_label = np.expand_dims(np.zeros(len(self.b.success_idx)), -1)
                observation_label = np.expand_dims(np.arange(0, len(self.b.success_idx), 1), -1)
                success_label = np.expand_dims(np.ones(len(self.b.success_idx)), -1)
                sort_idx = np.arange(0, len(self.b.success_idx), 1)
                return episode_label, observation_label, success_label, sort_idx
            episodes = np.concatenate([self.b.success_idx[:, 0], self.b.failures_idx[:, 0]])
            sort_idx = np.argsort(episodes)

            episodes_sorted = episodes[sort_idx]
            failure_start = self.b.failures_idx[:, 0]
            success_start = self.b.success_idx[:, 0]
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
        success_gaze = self.detectGazeEvents(self.b.success_idx, normalize=saccade_normalize, th_angle_p=25)
        success_fs = self.startForwardSwing(self.b.success_idx)
        success_start_fs = success_fs["start_fs"][:, 0] if len(self.b.success_idx) == 1 else success_fs[
            "start_fs"].squeeze()
        success_anglerac = self.angleRacketShoulder(self.b.success_idx, success_start_fs)
        success_mag_fs = self.speedACCForwardSwing(self.b.success_idx, success_start_fs)
        success_mag_impact = self.speedACCBeforeImpact(self.b.success_idx)

        features = np.hstack([success_gaze["gaze_event"],
                              success_gaze["saccade_p1"],
                              success_gaze["saccade_p2"],
                              success_gaze["fsp_phase3"],
                              success_fs["start_fs"] * 10,
                              success_anglerac["events_twoseg_angles"],
                              success_mag_fs["racket_speed"],
                              success_mag_fs["racket_acc"],
                              success_mag_fs["ball_speed"],
                              success_mag_fs["ball_acc"],
                              success_mag_fs["racket_speed"] / success_mag_fs["ball_speed"],
                              success_mag_fs["rball_dir"],
                              success_mag_fs["rball_dist"],
                              success_mag_impact["im_racket_force"],
                              success_mag_impact["im_ball_force"],
                              success_mag_impact["im_rb_ang_collision"],
                              success_mag_impact["im_ball_fimp"],
                              success_mag_impact["im_rb_dist"],
                              success_mag_impact["im_to_wrist_dist"],
                              success_mag_impact["im_rack_wrist_dist"],

                              ])
        # failures episode features
        if len(self.b.failures_idx):
            failures_gaze = self.detectGazeEvents(self.b.failures_idx, normalize=saccade_normalize, th_angle_p=25)
            failures_fs = self.startForwardSwing(self.b.failures_idx)
            failure_start_fs = failures_fs["start_fs"][:, 0] if len(self.b.failures_idx) == 1 else failures_fs[
                "start_fs"].squeeze()
            failures_anglerac = self.angleRacketShoulder(self.b.failures_idx, failure_start_fs)
            failures_mag_fs = self.speedACCForwardSwing(self.b.failures_idx, failure_start_fs)
            failures_mag_impact = self.speedACCBeforeImpact(self.b.failures_idx)
            failures_features = np.hstack([
                failures_gaze["gaze_event"],
                failures_gaze["saccade_p1"],
                failures_gaze["saccade_p2"],
                failures_gaze["fsp_phase3"],
                failures_fs["start_fs"] * 10,
                failures_anglerac["events_twoseg_angles"],
                failures_mag_fs["racket_speed"],
                failures_mag_fs["racket_acc"],
                failures_mag_fs["ball_speed"],
                failures_mag_fs["ball_acc"],
                failures_mag_fs["racket_speed"] / failures_mag_fs["ball_speed"],
                failures_mag_fs["rball_dir"],
                failures_mag_fs["rball_dist"],
                failures_mag_impact["im_racket_force"],
                failures_mag_impact["im_ball_force"],
                failures_mag_impact["im_rb_ang_collision"],
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

    def __init__(self, s1_features:FeaturesExtractor, s2_features:FeaturesExtractor, ball:Ball, table_wall:TableWall):

        self.s1 = s1_features
        self.s2 = s2_features
        self.b = ball
        self.table_wall = table_wall


    def extractGlobalFeatures(self, saccade_normalize=False):

        # s1_success_gaze = self.s1.detectGazeEvents(self.b.success_idx, normalize=saccade_normalize, th_angle_p=25)
        # s2_success_gaze = self.s2.detectGazeEvents(self.b.success_idx, normalize=saccade_normalize, th_angle_p=25)

        # joint_attention_success = self.s1.detectJointAttention(self.b.success_idx, self.s2)
        joint_attention_failure =  self.s1.detectJointAttention(self.b.failures_idx, self.s2)
        print("test")














