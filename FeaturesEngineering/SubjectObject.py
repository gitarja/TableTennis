from Utils.DataReader import TobiiReader
import numpy as np
from Utils.Lib import movingAverage



class Ball:

    def __init__(self, ball:dict):
        self.success_idx = ball["success_idx"]
        self.failures_idx = ball["failures_idx"]

        # remove failure episodes from success
        if len(self.failures_idx) > 0:
            self.success_idx = self.success_idx[~np.in1d(self.success_idx[:, 1], self.failures_idx[:, 1])]
            # Get previous episodes
            self.prev_s_idx = np.argwhere(np.isin(self.success_idx[:, 1], self.success_idx[:, 0]))
            self.prev_f_idx = np.argwhere(np.isin(self.success_idx[:, 1], self.failures_idx[:, 0]))
            self.all_episodes = np.vstack([self.success_idx, self.failures_idx])
        else:
            self.all_episodes = self.success_idx



    def smoothBall(self, ball: np.array) -> np.array:
        '''
        :param ball: ball in the world
        :return: smoothed ball trajectory
        '''

        ball = np.array([movingAverage(ball[:, i], n=1) for i in range(3)]).transpose()
        return ball

class Subject:


    def __init__(self, sub: dict, tobii: dict,  racket: dict, ball:Ball):
        # tobii reader
        self.tobii_reader = TobiiReader()
        self.ball = ball

        s = sub["segments"]
        r = racket["segments"]
        r_tj = racket["trajectories"]
        tobii = tobii["trajectories"]


        # ----------------------------------gaze direction-----------------------------
        self.eye_event = tobii.loc[:]["Eye_movement_type"].values.astype(float)
        self.tobii_time = tobii.loc[:]["Timestamp"].values.astype(float)
        tobii.iloc[tobii["Timestamp"] == 0, 1:] = np.nan
        self.gaze_point = tobii.filter(regex='Gaze_point_3D').values
        self.left_gaze_dir = tobii.filter(regex='Gaze_direction_left').values
        self.left_eye = tobii.filter(regex='Pupil_position_left').values
        self.right_eye = tobii.filter(regex='Pupil_position_right').values
        self.right_gaze_dir = tobii.filter(regex='Gaze_direction_right').values

        # ----------------------------------body-----------------------------
        self.root_segment_T = s.filter(regex='Root_T').values
        # wrist
        self.rwirst_segment_T = s.filter(regex='R_Wrist_T').values
        self.lwirst_segment_T = s.filter(regex='L_Wrist_T').values

        # colar
        self.rcollar_segment_T = s.filter(regex='R_Collar_T').values
        self.lcollar_segment_T = s.filter(regex='L_Collar_T').values

        self.rhummer_segment_T = s.filter(regex='R_Humerus_T').values
        self.relbow_segment_T = s.filter(regex='R_Elbow_T').values

        self.tobii_segment_T = s.filter(regex='TobiiGlass_T').values
        self.tobii_segment_R = s.filter(regex='TobiiGlass_R').values

        # get segments
        self.root_segment_T = s.filter(regex='Root_T').values
        # wrist
        self.rwirst_segment_T = s.filter(regex='R_Wrist_T').values
        self.lwirst_segment_T = s.filter(regex='L_Wrist_T').values

        # colar
        self.rcollar_segment_T = s.filter(regex='R_Collar_T').values
        self.lcollar_segment_T = s.filter(regex='L_Collar_T').values

        self.rhummer_segment_T = s.filter(regex='R_Humerus_T').values
        self.relbow_segment_T = s.filter(regex='R_Elbow_T').values

        self.racket_segment_T = r.filter(regex='pt_T').values
        self.racket_segment_R = r.filter(regex='pt_R').values
        self.racket_p3_T = r_tj.filter(regex='pt3_').values

        self.tobii_segment_T = s.filter(regex='TobiiGlass_T').values
        self.tobii_segment_R = s.filter(regex='TobiiGlass_R').values

        # ----------------------------------racket-----------------------------
        self.racket_segment_T = r.filter(regex='pt_T').values
        self.racket_segment_R = r.filter(regex='pt_R').values
        self.racket_p3_T = r_tj.filter(regex='pt3_').values

        self.normalizeTobii()


    def normalizeTobii(self):
        '''
        perform gap filling in Tobii
        :return:
        '''
        for e in self.ball.all_episodes:
            start = e[0] - 10
            stop = e[1] + 10

            self.left_gaze_dir[start:stop] = self.tobii_reader.gapFill(self.left_gaze_dir[start:stop])
            self.right_gaze_dir[start:stop] = self.tobii_reader.gapFill(self.right_gaze_dir[start:stop])

            self.gaze_point[start:stop] = self.tobii_reader.gapFill(self.gaze_point[start:stop])


