import matplotlib.pyplot as plt
import numpy as np
from Utils.Lib import cartesianToSpher


def gazeEventsPlotting(gaze, tobii_avg, ball, onset_offset_saccade, onset_offset_sp, onset_offset_fix, stream_label, al_cs_labels):
    _, gaze_az, gaze_elv = cartesianToSpher(vector=gaze - tobii_avg, swap=False)
    _, ball_az, ball_elv = cartesianToSpher(vector=ball - tobii_avg, swap=False)
    gaze_plot = np.vstack([gaze_az, gaze_elv]).transpose()
    ball_plot = np.vstack([ball_az, ball_elv]).transpose()
    x = gaze_plot[:, 0]
    y = gaze_plot[:, 1]
    x_ball = ball_plot[:-10, 0]
    y_ball = ball_plot[:-10, 1]

    plt.plot(x, y, linestyle="--", color="#525252")

    plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1.5, width=0.002, ls="dashed", color="#525252")

    plt.plot(x_ball, y_ball, linestyle="--", color="#000000")
    plt.quiver(x_ball[:-1], y_ball[:-1], x_ball[1:] - x_ball[:-1], y_ball[1:] - y_ball[:-1], scale_units='xy',
               angles='xy', scale=2., width=0.002, ls="dashed", color="#000000")

    for on, off in onset_offset_sp:
        plt.scatter(x[on:off + 1], y[on:off + 1], color="#4daf4a", zorder=2, marker="o", s=10)

    for on, off in onset_offset_fix:
        plt.scatter(x[on:off + 1], y[on:off + 1], color="#636363", zorder=3, s=10)

    for on_off, al in zip(onset_offset_saccade, al_cs_labels):
        on, off = on_off
        if al == 0:
            plt.scatter(x[on:off + 1], y[on:off + 1], color="#636363", zorder=4)

        elif al == 1:
            stream_label[on:off] = 4
            plt.scatter(x[on:off + 1], y[on:off + 1], color="#e41a1c", zorder=4)
        else:
            stream_label[on:off] = 5
            plt.scatter(x[on:off + 1], y[on:off + 1], color="#ff7f00", zorder=4)
    # print(len(x))
    # plt.savefig("F:\\users\\prasetia\\projects\\Animations\\TableTennis\\events.eps", format='eps')
    plt.xlim(-47, -9)
    plt.ylim(-53, -8)
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # color_labels = ["black", "blue", "red", "magenta", "yellow"]
    # ax.scatter(ball[:, 0], ball[:, 1], ball[:, 2], alpha=0.01, depthshade=False)
    # # ax.scatter(gaze[:, 0], gaze[:, 1], gaze[:, 2])
    #
    # for i in range(5):
    #     idx = np.argwhere(stream_label == (i + 1))
    #     ax.scatter(ball[idx, 0], ball[idx, 1], ball[idx, 2], color=color_labels[i], zorder=4,
    #                alpha=1, depthshade=False)
    #
    # plt.show()
