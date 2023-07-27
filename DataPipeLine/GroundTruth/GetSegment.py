from Utils.DataReader import ViconReader
import numpy as np
import ezc3d
import pandas as pd
from Utils.DataReader import TobiiReader
from Utils.Lib import wienerFilter, movingAverage, savgolFilter, cartesianToSpher
import matplotlib.pyplot as plt
def filteringUnLabData(data, tobii_data):
    # transform matrix
    data = data.transpose((-1, 1, 0))
    tobii_data = tobii_data.transpose((1, 0, -1))

    data = data[:len(tobii_data), :, :]

    tobii_dist_1 = np.linalg.norm(data - np.expand_dims(tobii_data[:, 0, :], 1), axis=-1)
    tobii_m = (tobii_dist_1 < 350)

    data[tobii_m] = np.nan

    ball_t = np.nanmean(data, axis=1)

    return ball_t


def matchTobii(cut_data, vicon_df, start):
    vicon_frame = len(vicon_df)
    # print(vicon_frame)
    selected_columns = ["Recording timestamp",
                        "Gaze point X", "Gaze point Y",
                        "Gaze point 3D X", "Gaze point 3D Y", "Gaze point 3D Z",
                        "Gaze direction left X", "Gaze direction left Y", "Gaze direction left Z",
                        "Gaze direction right X", "Gaze direction right Y", "Gaze direction right Z",
                        "Pupil position left X", "Pupil position left Y", "Pupil position left Z",
                        "Pupil position right X", "Pupil position right Y", "Pupil position right Z",
                        "Eye movement type"]

    # cut_data = cut_data.dropna(subset=["Gaze direction left X"])
    # print( 100 * np.average(np.isnan(cut_data["Gaze direction left X"].values)))

    columns = ["Timestamp",
               "Gaze_point_X", "Gaze_point_Y",
               "Gaze_point_3D_X", "Gaze_point_3D_Y", "Gaze_point_3D_Z",
               "Gaze_direction_left_X", "Gaze_direction_left_Y", "Gaze_direction_left_Z",
               "Gaze_direction_right X", "Gaze_direction_right_Y", "Gaze_direction_right_Z",
               "Pupil_position_left_X", "Pupil_position_left_Y", "Pupil_position_left_Z",
               "Pupil_position_right_X", "Pupil_position_right_Y", "Pupil_position_right_Z", "Eye_movement_type"]

    temp_data = np.empty((vicon_frame, len(columns)))
    tobii_df = pd.DataFrame(data=temp_data, columns=columns)
    start = start + np.arange(vicon_frame) * 10
    valid = (cut_data["Validity left"].values == "Valid") | (cut_data["Validity right"].values == "Valid")
    tobii_time = cut_data["Recording timestamp"].values
    dist_mat = np.abs(np.expand_dims(start, axis=0) - np.expand_dims(tobii_time, 1))
    closest_points = np.nanmin(dist_mat, axis=-1)
    closest_idx = np.nanargmin(dist_mat, axis=-1)
    selected_idx = np.argwhere((closest_points <= 5) & (valid))[:, 0]
    tobii_df.iloc[closest_idx[selected_idx]] = cut_data.iloc[selected_idx][
        selected_columns]


    return tobii_df

reader = ViconReader()
path = "F:\\users\\prasetia\\data\\TableTennis\\GT\\"
obj, sub, n = reader.extractData(path + "test76.csv", cleaning=True)

data = ezc3d.c3d(path + "test76.c3d")
labels = data['parameters']['POINT']['LABELS']['value']
unlabeled_idx = [i for i in range(len(labels)) if
                 "*" in labels[i]]  # the column label of the unlabelled marker starts with *
data_points = np.array(data['data']['points'])

unlabelled_data = data_points[0:3, unlabeled_idx, :]
tobii_T =sub[0]["segments"].filter(regex='TobiiGlass_T').values
tobii_R = sub[0]["segments"].filter(regex='TobiiGlass_R').values

ball_t = filteringUnLabData(unlabelled_data,  np.expand_dims(tobii_T, 0))


# read tobii data

tobii_data = pd.read_csv(path + "TobiiCoordinate_Recording_4.tsv", delimiter="\t")
start = 10222
stop = 66021
cut_data = tobii_data[
            (tobii_data["Recording timestamp"] >= start) & (tobii_data["Recording timestamp"] <= stop) & (
                    tobii_data["Sensor"] == "Eye Tracker")]


tobii = matchTobii(cut_data, tobii_T, start)

gaze_point = tobii.filter(regex='Gaze_point_3D').values
tobii_time = tobii.filter(regex='Timestamp').values


# analyze GT Global2Local
tobii_reader = TobiiReader()
ball_n_global = tobii_reader.global2LocalGaze(ball_t, tobii_T, tobii_R, translation=True)

_, b_az_g, b_inc_g = cartesianToSpher(ball_n_global, swap=True)
_, g_az_g, g_inc_g = cartesianToSpher(gaze_point, swap=True)

dist_angle_g = np.sqrt(np.square(b_az_g - g_az_g) + np.square(b_inc_g - g_inc_g))


# analyze GT local2Global
ball_n = ball_t - tobii_T
gaze_point_local  = tobii_reader.local2GlobalGaze(gaze_point, tobii_T, tobii_R, translation=True) - tobii_T

_, b_az_l, b_inc_l = cartesianToSpher(ball_n, swap=False)
_, g_az_l, g_inc_l = cartesianToSpher(gaze_point_local, swap=False)

dist_angle = np.sqrt(np.square(b_az_l - g_az_l) + np.square(b_inc_l - g_inc_l))
# s = 3129 # eye fixed
s = 1321 # tracking
plt.subplot(211)
plt.scatter(g_az_g[s:s+300], g_inc_g[s:s+300], color="blue")
plt.scatter(b_az_g[s:s+300], b_inc_g[s:s+300], color="red")
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.subplot(212)
plt.scatter(g_az_l[s:s+300], g_inc_l[s:s+300], color="blue")
plt.scatter(b_az_l[s:s+300], b_inc_l[s:s+300], color="red")
plt.xlim([-180, 180])
plt.ylim([-90, 90])
plt.show()

# fig, axs = plt.subplots(4, 5)
# k = 0
# for i in range(4):
#     for j in range(5):
#         print(tobii_time[s+k])
#
#         axs[i, j].scatter(g_az[s + k], g_inc[s + k], color="blue")
#         axs[i, j].scatter(b_az[s + k], b_inc[s + k], color="red")
#
#         # axs[i, j].axvline(x=0, color='b')
#         # axs[i, j].axhline(y=0, color='b')
#         axs[i, j].set_xlim([-180, 180])
#         axs[i, j].set_ylim([-90, 90])
#         axs[i, j].set_title(str(k))
#         k += 10
# plt.show()
# plt.close()