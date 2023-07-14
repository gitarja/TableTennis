from FeaturesExtractor.ClassicFeatures import Classic
from Utils.DataReader import SubjectObjectReader
from Utils.Visualization import Visualization
import numpy as np

# define readers
reader = SubjectObjectReader()
vis = Visualization()
paths = [
    "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-08_A\\2022-11-08_A_T01_complete.pkl",
    "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-08_A\\2022-11-08_A_T03_complete.pkl",
    "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-08_A\\2022-11-08_A_T04_complete.pkl",
    "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-09_A\\2022-11-09_A_T07_complete.pkl",
    "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-09_A\\2022-11-09_A_T04_complete.pkl",
    "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-09_A\\2022-11-09_A_T03_complete.pkl",
    "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2023-02-08_A\\2023-02-08_A_T02_complete.pkl",
    "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2023-02-08_A\\2023-02-08_A_T04_complete.pkl",
    "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2023-02-08_A\\2023-02-08_A_T03_complete.pkl"
]

# success features list
s_ball_impact_list = [] # success ball position at the moment of contact
s_ballDist_impact_list = [] # ||ball_curr-ball_prev|| at the moment of contact
s_acc_rc_list = [] # acceleration of racket before the impact (current)
s_acc_rp_list = [] # acceleration of racket before the impact (previous)
s_ag_brc_list = [] # angle between racket and angle at the impact (current)
s_fsc_list = [] # angle between racket and angle at the impact (current)
s_fsp_list = [] # angle between racket and angle at the impact (previous)
s_saccade_m_list = [] # angle between racket and angle at the impact
s_gba_list = [] # angle between gaze and ball before bounce

# failure features list
f_ball_impact_list = []# failure ball position at the moment of contact
f_ballDist_impact_list = [] # ||ball_curr-ball_prev|| at the moment of contact
f_acc_rc_list = [] # acceleration of racket before the impact
f_acc_rp_list = [] # acceleration of racket before the impact (previous)
f_ag_brc_list = [] # angle between racket and angle at the impact (current)
f_fsc_list = [] # angle between racket and angle at the impact (current)
f_fsp_list = [] # angle between racket and angle at the impact (previous)
f_saccade_m_list = [] # angle between racket and angle at the impact
f_gba_list = [] # angle between gaze and ball before bounce

for p in paths:
    print(p)
    obj, sub, ball, tobii = reader.extractData(p)
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
    features_extractor = Classic(sub[0], racket, ball[0], tobii[0], table, wall)

    # compute ball position at the contact moment
    # s_ball_impact, f_ball_impact = features_extractor.extractBallContactPosition(prev=False)
    # s_ball_impact_list.append(s_ball_impact["position"])
    # f_ball_impact_list.append(f_ball_impact["position"])

    # compute distance ball position at the contact moment (prev-current)
    # s_ball_impact_pc, f_ball_impact_pc = features_extractor.extractBallContactPosition(prev=True)
    # s_ballDist_impact_list.append(np.sqrt(np.square(s_ball_impact_pc["prev_position"] - s_ball_impact_pc["current_position"])))
    # f_ballDist_impact_list.append(np.sqrt(np.square(f_ball_impact_pc["prev_position"] - f_ball_impact_pc["current_position"])))

    # compute speed and acceleration before impact
    # s_acc_rc, f_acc_rc = features_extractor.extractRacketVelocity(1, prev=False)
    # s_acc_rc_list.append(s_acc_rc["speed"] * 10)
    # f_acc_rc_list.append(f_acc_rc["speed"]* 10)
    # s_acc_rp, f_acc_rp = features_extractor.extractRacketVelocity(1, prev=True)
    # s_acc_rp_list.append(s_acc_rp["speed"]* 10)
    # f_acc_rp_list.append(f_acc_rp["speed"] * 10)

    # compute angles
    # s_ag_brc, f_ag_brc = features_extractor.extractRacketBallAngleImpact(prev=False)
    # s_ag_brc_list.append(s_ag_brc["angle"])
    # f_ag_brc_list.append(f_ag_brc["angle"])

    # s_fsc, f_fsc = features_extractor.extractForwardswing(prev=False)
    # s_fsc_list.append(s_fsc["start_fs"])
    # f_fsc_list.append(f_fsc["start_fs"])
    #
    # s_fsp, f_fsp = features_extractor.extractForwardswing(prev=True)
    # s_fsp_list.append(s_fsp["start_fs"])
    # f_fsp_list.append(f_fsp["start_fs"])

    # compute gaze ball angle
    # s_gba, f_gba = features_extractor.extractGazeBallAngle(e_idx=3, n=20)
    # s_gba_list.append(s_gba["g_ba"])
    # f_gba_list.append(f_gba["g_ba"])
    # s_sacc_pursuit_c, f_sacc_pursuit_c = features_extractor.extractSaccadePursuit(normalize=False)
    # s_saccade_m_list.append(s_sacc_pursuit_c["p2_features"][:, 1])

# visualization

#visualize ball contact position
# vis.plotMultipleAxisHist(np.concatenate(s_ball_impact_list, 0) / 10, np.concatenate(f_ball_impact_list, 0) / 10, x_axis="cm")

#visualize ball contact position ||prev-current||
# vis.plotMultipleAxisHist(np.concatenate(s_ballDist_impact_list, 0) / 10, np.concatenate(f_ballDist_impact_list, 0) / 10, x_axis="||prev-current||(cm)")

#visualize racket acceleration
# vis.plotFourLines(np.concatenate(s_acc_rc_list, 0), np.concatenate(f_acc_rc_list, 0), np.concatenate(s_acc_rp_list, 0), np.concatenate(f_acc_rp_list, 0))

# vis.plotLine(np.concatenate(s_acc_rc_list, 0), np.concatenate(f_acc_rc_list, 0))

# visualize angle at the impact
vis.plotTwoHist(np.concatenate(s_ag_brc_list, 0), np.concatenate(f_ag_brc_list, 0))


#visualize start of forward swing
# vis.plotMultiCategori(np.concatenate(s_fsc_list, 0), np.concatenate(f_fsc_list, 0), np.concatenate(s_fsp_list, 0), np.concatenate(f_fsp_list, 0))


# visualize gaze ball angle
# vis.plotLine(np.concatenate(s_gba_list, 0), np.concatenate(f_gba_list, 0))