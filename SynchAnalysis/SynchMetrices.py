import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import welch, hilbert


def computeRelativePhase(v1, v2, n, m):
    v1_analytic = hilbert(v1)
    v2_analytic = hilbert(v2)
    v1_phase = np.angle(v1_analytic)
    v2_phase = np.angle(v2_analytic)

    relative_phase = ((n * v1_phase) - (m * v2_phase)) % (2 * np.pi)

    return relative_phase

def normalizedSignal(v1, v2,normalized =True, nt=0)->tuple:
    if normalized:
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
    if nt > 2:
        pad_num = nt - (v1.shape[1] - 1) % nt - 1

        # v1 = np.expand_dims(np.pad(v1[0], (pad_num // 2, pad_num //2), 'reflect'), 0)
        # v2 = np.expand_dims(np.pad(v2[0], (pad_num // 2, pad_num // 2), 'reflect'), 0)

        v1 = sliding_window_view(v1.flatten(), nt)
        v2 = sliding_window_view(v2.flatten(), nt)

    return v1, v2

def computeCompleteSync(v1, v2, normalized=True, nt=0):
    v1, v2 = normalizedSignal (v1, v2, normalized, nt)



    return 1 - np.average(np.abs((v1 - v2)),axis=-1)


def computePhaseSync(v1, v2, normalized=True, nt=0, n=1, m=1, return_phase=False):
    v1, v2 = normalizedSignal (v1, v2, normalized, nt)

    v1_analytic = hilbert(v1)
    v2_analytic = hilbert(v2)
    v1_phase = np.angle(v1_analytic)
    v2_phase = np.angle(v2_analytic)



    phase_diff = np.exp(1j * ( (n* v1_phase) -   (m * v2_phase)))

    avg_phase_diff = np.abs(np.average(phase_diff, axis=-1))
    print(avg_phase_diff)
    #
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(2)
    # axs[0].plot(v1[0], color='#252525')
    # axs[0].plot(v2[0], color='#e6550d')
    # axs[1].plot(v1_phase[0], color='#252525')
    # axs[1].plot(v2_phase[0], color='#e6550d')
    #
    # plt.show()


    if return_phase:
        return avg_phase_diff, n * v1_phase, m * v2_phase

    return avg_phase_diff


def computeLagSync(v1, v2, t, normalized=True, nt=0):

    v1 = v1[t:]
    v2 = v2[:-t]

    v1, v2 = normalizedSignal(v1, v2, normalized, nt)

    a = np.average(np.square(v2 - v1), -1)
    b = np.sqrt(np.average(np.square(v1), axis=-1) * np.average(np.square(v2), axis=-1))

    return np.sqrt(a / b)

def computePhaseLagSync(v1, v2,  nt=0, n=1, m=1, t=2):
    v1_analytic = hilbert(v1)
    v2_analytic = hilbert(v2)
    v1_phase = np.angle(v1_analytic)
    v2_phase = np.angle(v2_analytic)




    v1_phase = v1_phase[t:]
    v2_phase = v2_phase[:-t]


    if nt > 2:
        v1_phase = sliding_window_view(v1_phase, nt)
        v2_phase = sliding_window_view(v2_phase, nt)


    phase_diff = np.exp(1j * ( (n* v1_phase) -   (m * v2_phase)))

    avg_phase_diff = np.abs(np.average(phase_diff, axis=-1))


    return avg_phase_diff


def computeEntropySynchIndex(v1, v2, n_bins = 10, nt=0, n=1, m=1):

    v1, v2 = normalizedSignal (v1, v2, True, nt)

    relative_phase = computeRelativePhase(v1, v2, n, m)
    index = np.zeros(len(relative_phase))
    if np.sum(np.isnan(relative_phase)) == 0:
        for i in range(len(relative_phase)):
            p1, bins = np.histogram(relative_phase[i], bins=n_bins)

            p1 = p1 / len(relative_phase[i])
            non_zero_data = p1[p1 != 0]
            SMax = np.log(len(bins))
            S = - np.sum(non_zero_data * np.log(non_zero_data))

            ps = (SMax - S) / SMax

            index[i] = ps

    return index

def computeCondProbSynchIndex(v1, v2, n_bins = 10, n=1, m=1):
    v1_analytic = hilbert(v1)
    v2_analytic = hilbert(v2)
    v1_phase = np.angle(v1_analytic)
    v2_phase = np.angle(v2_analytic)


    p1, bins1 = np.histogram(v1_phase, bins=n_bins)
    p2, _ = np.histogram(v2_phase, bins=bins1)
    conditional_probs = []
    for i in range(len(bins1) - 1):
        mask = np.where((v1_phase >= bins1[i]) & (v1_phase <= bins1[i + 1]))[0]
        v2_segment = v2_phase[mask]
        # print("p1 %f, p2 %f" % (p1[i], np.abs(np.sum(np.exp(1j * v2_segment)))))
        conditional_probs.append(np.abs(np.sum(np.exp(1j * v2_segment)) / p1[i]))


    lmd = np.average(conditional_probs)
    return lmd


def computeFourierModeSynchIndex(v1, v2, n=1, m=1, nt=0):
    v1, v2 = normalizedSignal (v1, v2, True, nt)
    relative_phase = computeRelativePhase(v1, v2, n, m)

    index = np.zeros(len(relative_phase))
    for i in range(len(relative_phase)):
        lmd = np.square(np.average(np.cos(relative_phase[i]), axis=-1)) + np.square(np.average(np.sin(relative_phase[i]), axis=-1))
        index[i] = lmd

    return index


if __name__ == '__main__':
    v1 = generateSurrogateData(t="chaotic", n=10000)
    # v1 = np.cos(np.arange(0, 100, 1))
    # v2 = generateSurrogateData(t="random", n=100)
    v2 = generateIAAFTData(v1, nmax_iter=1500)

    # sync_score = computePhaseLagSync(v1,  v2, nt=20, t = 25)

    print(computePhaseSync(v1, v2))
    # import matplotlib.pyplot as plt
    #
    # plt.plot(sync_score)
    # plt.show()