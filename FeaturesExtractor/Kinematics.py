import numpy as np
from scipy.spatial.transform import Rotation as R


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
        yaw, pitch, roll = R.from_matrix(RM).as_euler("XYZ")
        return pitch, yaw, roll

    def computeSegmentOrientation(self, vectors1, vectors2):
        orientations = np.asarray([self.computeRotationAngles(v1, v2)for v1, v2 in zip(vectors1, vectors2)], dtype=float)

        return orientations, np.diff(orientations, n=1, axis=0), np.diff(orientations, n=2, axis=0)



    def computeSegmentAngles(self, v1, v2):
        v1_u = v1 / np.linalg.norm(v1)  # normalize v1
        v2_u = v2 / np.linalg.norm(v2)  # normalize v2
        angles = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        return angles, np.diff(angles, n=1), np.diff(angles, n=2)

    def computeSegmentDistances(self, v1, v2):
        distances = np.linalg.norm(v2 - v1)
        return distances, np.diff(distances, n=1), np.diff(distances, n=2)

    def extractFeatures(self, sub):
        s = sub["segments"]
        # get segments
        root_idx = (17 * 6) + 3
        lowerback_idx = (8 * 6) + 3
        rwrist_idx = (15 * 6) + 3
        rhumerus_idx = (13 * 6) + 3
        relbow_idx = (10 * 6) + 3
        rcolar_idx = (9 * 6) + 3

        root_segment = s.filter(regex='Root_T').values
        rwrist_segment = s.filter(regex='R_Wrist_T').values
        rhumerus_segment = s.filter(regex='R_Humerus_T').values
        relbow_segment = s.filter(regex='R_Elbow_T').values
        rcolar_segment = s.filter(regex='R_Collar_T').values

        forearm = rcolar_segment - rhumerus_segment
        upperarm = rwrist_segment - relbow_segment


        # get joints


        or_rw_r = self.computeSegmentOrientation(rwrist_segment, root_segment)
        or_rh_r = self.computeSegmentOrientation(rhumerus_segment, root_segment)
        or_re_r = self.computeSegmentOrientation(relbow_segment, root_segment)
        or_rc_r = self.computeSegmentOrientation(rcolar_segment, root_segment)
        or_up_fr = self.computeSegmentOrientation(upperarm, forearm)




if __name__ == '__main__':
    from Utils.DataReader import ViconReader

    reader = ViconReader()
    obj, sub, n_data = reader.extractData(
        "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\2022-11-08_A\\2022-11-08_A_T01.csv",
        cleaning=False)

    features_extractor = Kinematics()
    features_extractor.extractFeatures(sub[0])
