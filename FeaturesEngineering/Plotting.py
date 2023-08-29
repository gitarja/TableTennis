import matplotlib.pyplot as plt
import numpy as np
from Utils.Lib import cartesianToSpher


def gazeEventsPlotting(gaze, tobii_avg, ball, onset_offset_saccade, onset_offset_sp, onset_offset_fix, stream_label, al_cs_labels):
    _, gaze_az, gaze_elv = cartesianToSpher(vector=gaze - tobii_avg, swap=False)
    _, ball_az, ball_elv = cartesianToSpher(vector=ball, swap=False)
    gaze_plot = np.vstack([gaze_az, gaze_elv]).transpose()
    ball_plot = np.vstack([ball_az, ball_elv]).transpose()
    x = gaze_plot[:, 0]
    y = gaze_plot[:, 1]
    x_ball = ball_plot[:, 0]
    y_ball = ball_plot[:, 1]

    plt.plot(x, y, "-o")

    plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=2, width=0.002)

    # plt.plot(x_ball, y_ball, "-*")
    # plt.quiver(x_ball[:-1], y_ball[:-1], x_ball[1:] - x_ball[:-1], y_ball[1:] - y_ball[:-1], scale_units='xy',
    #            angles='xy', scale=2,
    #            width=0.002)

    for on, off in onset_offset_sp:
        plt.scatter(x[on:off + 1], y[on:off + 1], color="blue", zorder=2)

    for on, off in onset_offset_fix:
        plt.scatter(x[on:off + 1], y[on:off + 1], color="black", zorder=3)

    for on_off, al in zip(onset_offset_saccade, al_cs_labels):
        on, off = on_off
        if al == 0:
            plt.scatter(x[on:off + 1], y[on:off + 1], color="red", zorder=4)

        elif al == 1:
            stream_label[on:off] = 4
            plt.scatter(x[on:off + 1], y[on:off + 1], color="magenta", zorder=4)
        else:
            stream_label[on:off] = 5
            plt.scatter(x[on:off + 1], y[on:off + 1], color="yellow", zorder=4)

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    color_labels = ["black", "blue", "red", "magenta", "yellow"]
    ax.scatter(ball[:, 0], ball[:, 1], ball[:, 2], alpha=0.01, depthshade=False)
    # ax.scatter(gaze[:, 0], gaze[:, 1], gaze[:, 2])

    for i in range(5):
        idx = np.argwhere(stream_label == (i + 1))
        ax.scatter(ball[idx, 0], ball[idx, 1], ball[idx, 2], color=color_labels[i], zorder=4,
                   alpha=1, depthshade=False)

    plt.show()
