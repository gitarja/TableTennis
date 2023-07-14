import unittest
from FeaturesExtractor.ClassicFeatures import Classic
from Utils.DataReader import SubjectObjectReader
import numpy as np
class MyTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        reader = SubjectObjectReader()
        path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-08_A\\2022-11-08_A_T01_complete.pkl"
        obj, sub, ball, tobii = reader.extractData(path)
        racket = None
        table = None
        wall = None
        for o in obj:
            if "racket" in o["name"].lower():
                racket = o
            if "table" in o["name"].lower():
                table = o
            if "wall" in o["name"].lower():
                wall = o

        self.feature = Classic(sub[0], racket, ball[0], tobii[0], table, wall)

    def test_distanceBallBeforeEvent(self):
        ball = np.random.rand(200, 3)
        segment = np.random.rand(200, 3)
        episodes = np.array([[0, 10, 5, 7], [5, 20, 13, 15]])
        e_idx = 3
        n = 5
        ball_last = True
        features = self.feature.distanceBallBeforeEvent(ball, segment, episodes, e_idx, n, ball_last)
        self.assertEqual(features["d_before_event"].shape, (len(episodes), ))  # add assertion here
        self.assertEqual(features["d_axis"].shape, (len(episodes), 3))  # add assertion here
        features_phase = self.feature.distanceBallOverTime(ball, segment, episodes, e_idx, n)
        self.assertEqual(features_phase["d_ot_event"].shape, (2, n))
        features_gaze_ball = self.feature.gazeBallAngleBeforeEvent(episodes, e_idx, n)
        self.assertEqual(features_gaze_ball["g_b_ang"].shape, (2, n))


if __name__ == '__main__':
    unittest.main()
