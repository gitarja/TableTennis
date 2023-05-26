import numpy as np
from scipy.spatial.transform import Rotation as R
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import  matplotlib.pyplot as plt
from Utils.Lib import wienerFilter, movingAverage, savgolFilter
class Kinematics:



    def computeRotationMatrix(self, v1, v2):
        v1_u = v1 / np.linalg.norm(v1)  # normalize v1
        v2_u = v2 / np.linalg.norm(v2)  # normalize v2
        v = np.cross(v1_u, v2_u)
        s = np.linalg.norm(v)
        c = np.dot(v1_u, v2_u)
        v1, v2, v3 = v

        kmat = np.array([[0, -v3, v2]
                            , [v3, 0, -v1]
                            , [-v2, v1, 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

        return rotation_matrix


    def computeTranslationMatrix(self, v1, v2):
        t = v2 - v1
        T = np.identity(4)
        T[0:3, 3] = t
        return T

    def computeRotationAngles(self, v1, v2):
        RM = self.computeRotationMatrix(v1, v2)
        yaw, pitch, roll = R.from_matrix(RM).as_euler("xyz")
        return pitch, yaw, roll

    def computeSegmentOrientation(self, vectors1, vectors2):
        orientations = np.asarray([self.computeRotationAngles(v1, v2)for v1, v2 in zip(vectors1, vectors2)], dtype=float)

        return orientations, np.diff(orientations, n=1, axis=0), np.diff(orientations, n=2, axis=0)

    def computeSegmentOrientDiff(self, v1, v2):

        return v2 - v1

    def computeSegmentAngles(self, v1, v2):
        v1_u = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)   # normalize v1
        v2_u = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)   # normalize v2
        angles = np.arccos(np.clip(np.einsum('ij,ij->i', v1_u, v2_u), -1.0, 1.0))
        return angles, np.diff(angles, n=1), np.diff(angles, n=2)

    def computeSegmentDistances(self, v1, v2):
        distances = np.linalg.norm(v2 - v1, axis=-1)
        return distances, np.diff(distances, n=1), np.diff(distances, n=2)

    def computeVelAcc(self, v):
        # v1 = v[np.arange(1, len(v)-2, 1)]
        # v2 = v[np.arange(3, len(v), 1)]
        v1 = v[:-1]
        v2 = v[1:]
        # return  np.sum(np.abs(v2 - v1), axis=-1), np.sum(np.diff(v, n=2), axis=-1)
        return np.sum(np.abs(np.diff(v, n=1, axis=0)), axis=-1), np.sum(np.diff(np.abs(np.diff(v, n=1, axis=0)), n=1, axis=0), axis=-1)


    def smoothBall(self, ball, success_idx, failures_idx):
        # offset = 3
        # for s in success_idx:
        #     ball[s[0]-offset: s[1] + offset] = np.array([movingAverage(ball[s[0]-offset: s[1] + offset, i], n=3) for i in range(3)]).transpose()

        # for f in failures_idx:
        #     ball[f[0] - offset: f[1] + offset] = wienerFilter(ball[f[0] - offset: f[1]+offset], n=1)

        ball= np.array([movingAverage(ball[:, i], n=2) for i in range(3)]).transpose()
        return ball

    def extractFeatures(self, sub, racket, ball):
        s = sub["segments"]
        r = racket["segments"]

        success_idx = ball["success_idx"]
        failures_idx = ball["failures_idx"]
        # get segments

        root_segment_T = s.filter(regex='Root_T').values
        rwrist_segment_T = s.filter(regex='R_Wrist_T').values
        rhumerus_segment_T = s.filter(regex='R_Humerus_T').values
        relbow_segment_T = s.filter(regex='R_Elbow_T').values
        rcolar_segment_T = s.filter(regex='R_Collar_T').values

        root_segment_R = s.filter(regex='Root_R').values
        rwrist_segment_R = s.filter(regex='R_Wrist_R').values
        rhumerus_segment_R = s.filter(regex='R_Humerus_R').values
        relbow_segment_R = s.filter(regex='R_Elbow_R').values
        rcolar_segment_R = s.filter(regex='R_Collar_R').values

        racket_segment = r.filter(regex='pt_T').values
        racket_segment_R = r.filter(regex='pt_R').values

        # get trajectories
        ball_t = self.smoothBall(ball["trajectories"].values, success_idx, failures_idx)

        ball_racket = self.computeSegmentAngles(racket_segment - root_segment_T, ball_t - racket_segment)
        racket_vel, racket_acc = self.computeVelAcc(racket_segment)
        ball_vel, ball_acc = self.computeVelAcc(ball_t)
        rw_vel, rw_acc = self.computeVelAcc(rwrist_segment_T)

        # r_rw = self.computeSegmentAngles(root_segment_T, rwrist_segment_T)[1]

        # success_dtw = []
        # failures_dtw = []
        # for i in range(5):
        #     s1 = success_idx[i]
        #     s2 = success_idx[(i)+1]
        #     success_dtw.append(dtw.distance(r_rw[s1[0]:s1[1]], r_rw[s2[0]:s2[1]]))

        # for i in range(4):
        #     f2 = failures_idx[i]
        #     f1 = success_idx[np.argwhere(success_idx[:, 1] == f2[0])][0, 0]
        #     failures_dtw.append(dtw.distance(r_rw[f1[0]:f1[1]], r_rw[f2[0]:f2[1]]))

        # dtw_dist = []
        # f2 = failures_idx[0]
        # s_idx = np.argwhere(success_idx[:, 1] == f2[0])[0,0]
        # for i in range(s_idx+1):
        #     s1 = success_idx[i]
        #     s2 = success_idx[(i)+1]
        #     dtw_dist.append(dtw.distance(r_rw[s1[0]:s1[1]], r_rw[s2[0]:s2[1]]))
        #
        # f2 = failures_idx[0]
        # f1 = success_idx[s_idx]
        # dtw_dist.append(dtw.distance(r_rw[f1[0]:f1[1]], r_rw[f2[0]:f2[1]]))
        # plt.plot(dtw_dist, '--bo')
        # plt.show()

        # plt.boxplot([success_dtw, failures_dtw])
        #
        #
        # plt.show()



        import matplotlib.pyplot as plt


        y_lim = [-0.03, 0.03]
        br = ball_racket[0]



        success_start = []
        success_end = []

        failure_start = []
        failure_end = []

        for s in success_idx:
            # s = (s / 10).astype(int)
            bv = ball_vel[s[0]-1:s[1]]
            rv = racket_vel[s[0]-1:s[1]]

            print(bv[0])
            print(bv[-1])
            success_start.append(np.abs(bv[0] - rv[0]))
            success_end.append(np.abs(bv[-1] - rv[-1]))

        for s in failures_idx:
            # s = (s / 10).astype(int)
            bv = ball_vel[s[0]:s[1]]
            rv = racket_vel[s[0]:s[1]]

            failure_start.append(np.abs(bv[0] - rv[0]))
            failure_end.append(np.abs(bv[-1] - rv[-1]))


        plt.boxplot([success_start, failure_start, success_end, failure_end])
        plt.show()


        # fig, (ax1, ax3) = plt.subplots(2)
        # s = success_idx[5]
        # ax1.plot(ball_t[s[0]:s[1], 0], color="#e41a1c")
        # ax1.plot(ball_t[s[0]:s[1], 1], color="#e41a1c")
        # ax1.plot(ball_t[s[0]:s[1], 2], color="#e41a1c")
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        #
        # ax.scatter(ball_t[s[0]:s[1], 0], ball_t[s[0]:s[1], 1], ball_t[s[0]:s[1], 2])
        # plt.show()
        # ax1.plot(racket_vel[s[0]:s[1]], color="#377eb8")
        # ax1.axvline(s[2] - s[0], color='y', linestyle = 'dashed')
        # ax1.axvline(s[3]- s[0], color='g', linestyle = 'dashed')
        # ax1.set_ylabel('Ball velocity')
        #
        # s = success_idx[5]
        # ax3.plot(ball_vel[s[0]:s[1]], color="#e41a1c")
        # ax3.plot(racket_vel[s[0]:s[1]], color="#377eb8")
        # ax3.plot(rw_vel[s[0]:s[1]], color="#4daf4a")
        # ax3.axvline(s[2] - s[0], color='y', linestyle = 'dashed')
        # ax3.axvline(s[3] - s[0], color='g', linestyle = 'dashed')
        # ax3.set_ylabel('Ball velocity')
        #
        #
        # plt.show()

        # for f in failures_idx:
        #     plt.plot(ball_racket[1][f[0]:f[1]], color="red")
        # plt.show()
        #
        # print("Test")
        # get angle between orientation
        # or_rw_ra = self.computeSegmentOrientation(root_segment_R, rwrist_segment_R)
        # or_rh_ra = self.computeSegmentOrientation(root_segment_R, rhumerus_segment_R)
        # or_el_ra = self.computeSegmentOrientation(root_segment_R, relbow_segment_R)
        # or_cl_ra = self.computeSegmentOrientation(root_segment_R, rcolar_segment_R)
        #
        #
        # forearm = rcolar_segment_T - rhumerus_segment_T
        # upperarm = rwrist_segment_T - relbow_segment_T






if __name__ == '__main__':
    from Utils.DataReader import SubjectObjectReader

    reader = SubjectObjectReader()
    obj, sub, ball = reader.extractData("F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-08_A\\2022-11-08_A_T04_wb.pkl")

    features_extractor = Kinematics()
    features_extractor.extractFeatures(sub[0], obj[1], ball[0])
