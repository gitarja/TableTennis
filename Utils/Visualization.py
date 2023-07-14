import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.despine(offset=10, trim=True, left=True)
sns.set_palette(sns.color_palette("Set2"))

class Visualization:

    def plot3D(self, success, failures):
        success_idx = np.random.choice(len(success), len(failures))
        success = success[success_idx]

        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(success[:, 0], success[:, 1], success[:, 2])
        ax.scatter(failures[:, 0], failures[:, 1], failures[:, 2])

        plt.show()

    def plotWallHist(self, success_bounce, fail_bounces, wall):
        success_label = ["success"] * len(success_bounce)
        fail_label = ["failures"] * len(fail_bounces)

        bouces = np.vstack([success_bounce, fail_bounces])
        labels = np.hstack([success_label, fail_label])
        df = pd.DataFrame({"x": bouces[:, 0], "y": bouces[:, 1], "label": labels})
        sns.jointplot(data=df, x="x", y="y", hue="label", type="kde")
        # gfg = sns.kdeplot(
        #     data=df, x="x", y="y", hue="label", fill=True,
        # )
        # gfg.set_ylim(0, 80)
        plt.show()

    def plotTwoHist(self, success_angle, fail_angle, y="angles(radian)"):
        np.random.seed(2023)
        success_idx = np.random.choice(len(success_angle), len(fail_angle))
        success_angle = success_angle[success_idx]
        success_label = ["success"] * len(success_angle)
        fail_label = ["failures"] * len(fail_angle)

        angles = np.hstack([success_angle, fail_angle])
        labels = np.hstack([success_label, fail_label])
        df = pd.DataFrame({y: angles, "label": labels})

        # g = sns.displot(df,  x=y, hue="label", kind="kde", fill=True)
        # g = sns.catplot(df, y=y, x="label", kind="violin", split=True)
        g = sns.boxplot(x='label', y=y, data=df, whis=[0, 100], width=.6)
        # Add in points to show each observation
        sns.stripplot(x="label", y=y, data=df,
                      size=4, color=".3", linewidth=0)

        g.set(xlabel=None)
        g.set(ylabel=None)
        # g._legend.remove()
        # plt.savefig('F:\\users\\prasetia\\projects\\Animations\\Poster-2023-08-05\\angles.eps', format='eps')

        plt.show()

    def plotMultipleHist(self, success_ball, success_racket, fail_ball, fail_racket):
        np.random.seed(2025)
        success_idx = np.random.choice(len(success_ball), len(fail_ball))
        success_ball = success_ball[success_idx]

        success_idx = np.random.choice(len(success_racket), len(fail_racket))
        success_racket = success_racket[success_idx]

        types = (["ball"] * len(success_ball)) + (["ball"] * len(fail_ball)) + (["racket"] * len(success_racket)) + (
                ["racket"] * len(fail_racket))
        labels = (["success"] * len(success_ball)) + (["fail"] * len(fail_ball)) + (
                ["success"] * len(success_racket)) + (["fail"] * len(fail_racket))
        acc = np.hstack([success_ball, fail_ball, success_racket, fail_racket])

        df = pd.DataFrame({"acceleration": acc, "type": types, "label": labels})
        g = sns.catplot(
            data=df, x="type", y="acceleration", hue="label",
            kind="violin", split=True,
        )

        g.set(xlabel=None)
        g.set(ylabel=None)
        g._legend.remove()
        plt.savefig('F:\\users\\prasetia\\projects\\Animations\\Poster-2023-08-05\\ball_racket_acc.eps', format='eps')

        plt.show()

    def plotMultipleAxisHist(self, success_data, failure_data, x_axis="distance"):

        types = (["x", "y", "z"] * len(success_data)) + (["x", "y", "z"] * len(failure_data))
        labels = (["success"] * (3 * len(success_data))) + (["fail"] * (3 * len(failure_data)))
        dist = np.hstack([success_data.flatten(), failure_data.flatten()])

        df = pd.DataFrame({x_axis: dist, "axis": types, "label": labels})
        sns.catplot(
            data=df, x=x_axis, y="axis", hue="label",
            kind="violin", split=True
        )


        # plt.savefig('F:\\users\\prasetia\\projects\\Animations\\Poster-2023-08-05\\distance_axis.eps', format='eps')
        plt.show()

    def plotLine(self, success_y, fail_y):
        np.random.seed(2025)

        success_idx = np.random.choice(len(success_y), len(fail_y))
        success_y = success_y[success_idx]

        n = success_y.shape[-1]
        n_2 = int(n / 2)

        time_point = (np.arange(-n, 0).astype(str).tolist() * len(success_y)) + (
                    np.arange(-n, 0).astype(str).tolist() * len(fail_y))

        # time_point = (np.arange(-n_2, n_2).astype(str).tolist() * len(success_y)) + (
        #         np.arange(-n_2, n_2).astype(str).tolist() * len(fail_y))
        labels = (["success"] * (n * len(success_y))) + (["fail"] * (n * len(fail_y)))

        data = np.concatenate([success_y.flatten(), fail_y.flatten()])
        df = pd.DataFrame({"time_point": [float(i) * 10 for i in time_point], "acceleration_racket": data, "label": labels})

        g = sns.lineplot(data=df, x="time_point", y="acceleration_racket", hue="label", estimator="mean")
        sns.despine(offset=10, trim=True, left=True)
        # plt.vlines(0, 0, 10, linestyles="dotted", colors="r")
        # g.set_ylim([0, 5])
        g.set(xlabel=None)
        g.set(ylabel=None)


        # plt.savefig('F:\\users\\prasetia\\projects\\Animations\\Poster-2023-08-05\\accelerate_decelarate.eps', format='eps')
        plt.show()

    def plotFourLines(self, success_y, fail_y, prev_success_y, prev_fail_y):
        np.random.seed(2023)

        success_idx = np.random.choice(len(success_y), len(fail_y))
        success_y = success_y[success_idx]

        prev_success_idx = np.random.choice(len(prev_success_y), len(prev_fail_y))
        prev_success_y = prev_success_y[prev_success_idx]

        n = success_y.shape[-1]
        n_2 = int(n / 2)

        # time_point = (np.arange(-n, 0).astype(str).tolist() * len(success_y)) + \
        #              (np.arange(-n, 0).astype(str).tolist() * len(fail_y)) +\
        #              (np.arange(-n, 0).astype(str).tolist() * len(prev_success_y)) + \
        #              (np.arange(-n, 0).astype(str).tolist() * len(prev_fail_y))

        time_point = (np.arange(-n_2, n_2).astype(str).tolist() * len(success_y)) + (
                np.arange(-n_2, n_2).astype(str).tolist() * len(fail_y)) + (
                             np.arange(-n_2, n_2).astype(str).tolist() * len(prev_success_y)) + (
                             np.arange(-n_2, n_2).astype(str).tolist() * len(prev_fail_y))
        labels = (["success"] * (n * len(success_y))) + (["fail"] * (n * len(fail_y))) + (
                ["pre_success"] * (n * len(prev_success_y))) + \
                 (["prev_fail"] * (n * len(prev_fail_y)))

        data = np.concatenate([success_y.flatten(), fail_y.flatten(), prev_success_y.flatten(), prev_fail_y.flatten()])

        df = pd.DataFrame({"time_point": [float(i) * 10 for i in time_point], "acceleration_racket": data, "label": labels})

        g = sns.lineplot(data=df, x="time_point", y="acceleration_racket", hue="label", estimator="mean")
        sns.despine(offset=10, trim=True, left=True)

        g.set(xlabel=None)
        g.set(ylabel=None)

        # plt.savefig('F:\\users\\prasetia\\projects\\Animations\\Poster-2023-08-05\\accelerate_decelarate.eps', format='eps')
        plt.show()

    def plotLine2(self, success_y, fail_y, success_event, failure_event):
        np.random.seed(2023)

        ev = -1 * int(np.average(np.concatenate([success_event, failure_event])))

        success_idx = np.random.choice(len(success_y), len(fail_y))
        success_y = success_y[success_idx]
        # fail_y = fail_y[:, ev:]

        n = success_y.shape[-1]

        time_point = (np.arange(-n, 0).astype(str).tolist() * len(success_y)) + (
                np.arange(-n, 0).astype(str).tolist() * len(fail_y))
        labels = (["success"] * (n * len(success_y))) + (["fail"] * (n * len(fail_y)))

        data = np.concatenate([success_y.flatten(), fail_y.flatten()])
        df = pd.DataFrame({"time_point": [float(i) for i in time_point], "acceleration_racket": data, "label": labels})

        g = sns.lineplot(data=df, x="time_point", y="acceleration_racket", hue="label")
        sns.despine(offset=10, trim=True, left=True)
        g.set(xlabel=None)
        g.set(ylabel=None)
        g.legend().remove()
        plt.vlines(int(ev), -1, 500, linestyles="dotted", colors="r")
        # plt.vlines(int(se), -1, 500, linestyles="dotted", colors="r")
        # plt.vlines(int(fe), -1, 500, linestyles="dotted", colors="g")
        # rect_se = mpatches.Rectangle((se, -1), se_w, 500,
        #                           # fill=False,
        #                           alpha=0.1,
        #                           facecolor="green")
        #
        # rect_fe = mpatches.Rectangle((fe, -1), fe_w, 500,
        #                           # fill=False,
        #                           alpha=0.1,
        #                           facecolor="red")
        # plt.gca().add_patch(rect_se)
        # plt.gca().add_patch(rect_fe)
        # plt.savefig('F:\\users\\prasetia\\projects\\Animations\\Poster-2023-08-05\\acc.eps', format='eps')
        plt.show()

    def plotMultiCategori(self, success, fail, prev_success, prev_fail):
        np.random.seed(2023)
        success_idx = np.random.choice(len(success), len(fail))
        success = success[success_idx]

        prev_success_idx = np.random.choice(len(prev_success), len(prev_fail))
        prev_success = prev_success[prev_success_idx]

        print(np.average(success))
        print(np.average(fail))
        print(np.average(prev_success))
        print(np.average(prev_fail))
        labels = (["success"] * len(success)) + (["fail"] * len(fail)) + (["prev_success"] * len(prev_success)) + (
                ["prev_fail"] * len(prev_fail))

        data = np.concatenate([success.flatten(), fail.flatten(), prev_success.flatten(), prev_fail.flatten()])

        df = pd.DataFrame({"RT": data, "label": labels})

        g = sns.boxplot(x='label', y='RT', data=df, whis=[0, 100], width=.6)
        # Add in points to show each observation
        sns.stripplot(x="label", y="RT", data=df,
                      size=4, color=".3", linewidth=0)
        # g = sns.displot(x='RT', hue='label', data=df, kind="kde", fill=True)

        sns.despine(offset=10, trim=True, left=True)
        g.set(xlabel=None)
        g.set(ylabel=None)

        plt.show()

    def plotScatter(self, success, fail, n=20, step=2):
        # success_idx = np.random.choice(len(success), len(fail))
        # success = success[success_idx]

        labels = (["success"] * len(success) * 20)
        time_point = (np.arange(-n, n, step).astype(str).tolist() * len(success))

        data = np.concatenate([success.reshape((-1, 2))])
        df = pd.DataFrame({"time_point": [float(i) for i in time_point], "azimuth": data[:, 0], "elevation": data[:, 1],
                           "label": labels})

        g = sns.FacetGrid(df, col="time_point", hue="label", col_wrap=5, height=2, ylim=(-45, 45), xlim=(-60, 60))
        g.map(sns.scatterplot, "azimuth", "elevation", alpha=0.5)
        # g.map(sns.kdeplot, "azimuth", "elevation", alpha=0.5, fill=True)
        g.refline(x=0,
                  y=0,
                  color="black",
                  lw=1)
        g.refline(x=5,
                  y=5,
                  color="red",
                  lw=1)
        g.refline(x=-5,
                  y=-5,
                  color="red",
                  lw=1)

        plt.show()