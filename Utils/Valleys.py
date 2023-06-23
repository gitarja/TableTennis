import numpy as np
from scipy.ndimage import label


def findValleys(dist:np.array, th_c:float, th_d:float) -> np.array:
    '''
    Detect valleys by performing template maching

    :param dist: distance between two objects (e.g. racket and wall)
    :param th_c: only consider maching score higher than th_c
    :param th_d: only consider valley with a height less than
    :return: index of the valleys
    '''
    # find the peak with template matching, create the cos template
    t = np.linspace(-1.5 * np.pi, -0.5 * np.pi, 3)
    qrs_filter = np.cos(t)

    # normalize data
    dist_norm = (dist - np.nanmean(dist)) / np.nanstd(dist)

    # calculate cross correlation
    similarity = np.correlate(dist_norm, qrs_filter, mode="same")
    similarity = similarity / np.nanmax(similarity)

    valleys = np.nonzero((similarity > th_c) & (dist <= th_d))[0]

    return valleys


def removeSpecialValleyTable(v_table:np.array, v_racket: np.array, th:int=5):
    removed_idx = []
    for v in v_racket:
        dist = v_table - v
        idx = np.argwhere((dist >= -th)& (dist<=2) )
        if len(idx) > 0:
            removed_idx.append(idx)

    if len(removed_idx) == 0:
        return v_table
    removed_idx = np.vstack(removed_idx)

    return np.delete(v_table, removed_idx)


def groupValleys(v:np.array, dist:np.array, within_th:int=10, n_group=(1, 4))-> np.array:

    """
    Valley detection algorithm to finds multiple valley.
    Group collections of valleys that are very near (within threshold) and we take the min index

    :param v: valleys index
    :param dist: distance between two objects (e.g. racket and wall)
    :param within_th: the within threshold
    :param n_group:
    :return: grouped valleys index
    """

    # initialize output
    output = np.empty(0)

    # pad array
    v = np.pad(v, 2, mode="edge")
    v = np.unique(np.sort(np.concatenate([v-1,v,v+1])))
    # label groups of sample that belong to the same peak
    valley_groups, num_groups = label(np.diff(v) < within_th)

    # iterate through groups and take the mean as peak index
    for i in np.unique(valley_groups)[1:]:
        valley_group = v[np.where(valley_groups == i)]
        # output = np.append(output, valley_group[np.argmin(dist[valley_group])])
        output = np.append(output, valley_group[np.argmin(dist[valley_group])])
        # if (len(valley_group) >= n_group[0]) & (len(valley_group) <= n_group[1]):
        #     output = np.append(output, valley_group[np.argmin(dist[valley_group])])
    return output


def checkValleysSanity(racket_vy:np.array, wall_vy:np.array, dis_th=100)->np.array:
    '''
    Check the sanity of the valley. Two racket valleys in different episode must bigger than dis_th.
    Two racket valleys in the same episode muss have wall valley (a time when the ball hits the wall)

    :param racket_vy: Racket valley index
    :param wall_vy: Wall valley index
    :param dis_th: the minimum distance between two episodes
    :return: cleaned racket valley index
    '''
    sane_valleys = []
    prev = 0
    for i in range(len(racket_vy) - 1):
        dist_r_r = np.abs(prev - racket_vy[i])
        # print(str(prev) + " " + str(racket_vy[i]))
        prev = racket_vy[i]
        if dist_r_r > dis_th:
            if np.sum((wall_vy > racket_vy[i]) & (wall_vy < racket_vy[i + 1])) >= 1:
                sane_valleys.append(racket_vy[i])
        else:
            sane_valleys.append(racket_vy[i])

    sane_valleys.append(racket_vy[len(racket_vy) - 1])

    return np.array(sane_valleys)
