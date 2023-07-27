import numpy as np
from scipy.signal import lfilter, convolve

def computeVelAccV2(v: np.array, normalize=True)-> np.array:
    v1 = v[1:]
    v2 = v[:-1]
    kernel_vel = np.array([0, 1, 2, 3, 2, 1, 0]) #memory and prediction in natural gaze
    velocity = computeSegmentAngles(v1, v2) * 100
    velocity = np.pad(velocity, (0, 1), 'symmetric')
    velocity_norm = lfilter(kernel_vel, 10, velocity)
    # velocity_norm = convolve(velocity, kernel_vel/10 , "same")
    if normalize:
        acceler = np.diff(velocity_norm, n=1, axis=0)
    else:
        acceler = np.diff(velocity, n=1, axis=0)
    acceler = np.pad(acceler, (0, 1), 'symmetric')

    import matplotlib.pyplot as plt

    # print(len(v))
    # print(len(velocity_norm))
    # print(len(acceler))
    # plt.plot(velocity)
    # plt.plot( lfilter(kernel_vel, 10, velocity))
    # plt.plot(np.convolve(velocity, kernel_vel/10, "same"))
    # plt.show()

    return velocity, velocity_norm, acceler

def computeVectorsDirection(v1: np.array, v2: np.array):
    v1_u = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)  # normalize v1
    v2_u = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)  # normalize v2
    direction = np.einsum('ij,ij->i', v1_u, v2_u)
    return direction

def computeSegmentAngles(v1: np.array, v2: np.array):
    v1_u = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)  # normalize v1
    v2_u = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)  # normalize v2
    angles = np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i', v1_u, v2_u), -1.0, 1.0)))
    return angles


def computeHistBouce(ball: np.array, episodes: np.array):
    wall_bounce = []
    table_bounce = []
    for e in episodes:
        wall_bounce.append(ball[e[2], [0, 2]])
        table_bounce.append(ball[e[3], [0, 1]])

    return np.vstack(wall_bounce), np.vstack(table_bounce)


def computeVelAcc(v):
    v1 = v[:-1]
    v2 = v[1:]
    speed = np.linalg.norm(v2 - v1, axis=-1)
    vel = np.sum(np.diff(v, n=1, axis=0), axis=-1)
    acc = np.diff(speed, n=1, axis=-1)

    return speed, vel, acc


# if __name__ == '__main__':
#     v1 = np.random.random((30))
#     v2 = np.random.random((30)) + 10
#     v3 = np.random.random((30)) + 20
#
#     v = np.concatenate([v1, v2, v3])
#
#     vel, acc, saccade = computeVelAccV2(v)
#     import matplotlib.pyplot as plt
#
#     plt.plot(saccade)
#     plt.show()
#     print(vel.shape)