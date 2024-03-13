double_features_col = [
    "session_id",
    "id_subject1",
    "skill_subject1",
    "id_subject2",
    "skill_subject2",
    "team_skill",
    "team_max_seq",
    "team_avg_seq",

    # gaze event of receiver
    "receiver_pr_p1_al",
    "receiver_pr_p2_al",
    "receiver_pr_p3_fx",
    "receiver_pr_p1_cs",
    "receiver_pr_p2_cs",
    "receiver_pr_p3_cs",
    "receiver_pr_p1_al_onset",
    "receiver_pr_p1_al_prec",
    "receiver_pr_p1_al_mag",
    "receiver_pr_p2_al_onset",
    "receiver_pr_p2_al_prec",
    "receiver_pr_p2_al_mag",
    "receiver_pr_p3_fx_onset",
    "receiver_pr_p3_fx_duration",
    "receiver_pr_p3_stability",

    # gaze event of hitter
    "hitter_pr_p1_al",
    "hitter_pr_p2_al",
    "hitter_pr_p3_fx",
    "hitter_pr_p1_cs",
    "hitter_pr_p2_cs",
    "hitter_pr_p3_cs",
    "hitter_pr_p1_al_onset",
    "hitter_pr_p1_al_prec",
    "hitter_pr_p1_al_mag",
    "hitter_pr_p2_al_onset",
    "hitter_pr_p2_al_prec",
    "hitter_pr_p2_al_mag",
    "hitter_pr_p3_fx_onset",
    "hitter_pr_p3_fx_duration",
    "hitter_pr_p3_stability",

    # Joint attention
    "ja_p23_minDu",
    "ja_p23_maxDu",
    "ja_p23_avgDu",
    "ja_p23_percentage",



    # Forward swing and impact
    "receiver_ec_start_fs",
    "receiver_ec_fs_ball_racket_ratio",
    "receiver_ec_fs_ball_racket_dir",
    "receiver_ec_fs_ball_rball_dist",
    "receiver_im_racket_force",
    "receiver_im_ball_force",
    "receiver_im_rb_ang_collision",
    "receiver_im_rb_dist",
    "receiver_im_rack_wrist_dist",


    # bouncing point
    "hitter_hit_to_bouncing_point",
    "receiver_racket_to_root",
    "team_spatial_position",  # only for double




    "hitter",
    "observer",
    "pair_idx",
    "episode_label",
    "observation_label",
    "success"


]


double_features_col_num = [

    "skill_subject1",
    "skill_subject2",
    "team_skill",
    "team_max_seq",
    "team_avg_seq",

    # gaze event of receiver
    "receiver_pr_p1_al",
    "receiver_pr_p2_al",
    "receiver_pr_p3_fx",
    "receiver_pr_p1_cs",
    "receiver_pr_p2_cs",
    "receiver_pr_p3_cs",
    "receiver_pr_p1_al_onset",
    "receiver_pr_p1_al_prec",
    "receiver_pr_p1_al_mag",
    "receiver_pr_p2_al_onset",
    "receiver_pr_p2_al_prec",
    "receiver_pr_p2_al_mag",
    "receiver_pr_p3_fx_onset",
    "receiver_pr_p3_fx_duration",
    "receiver_pr_p3_stability",

    # gaze event of hitter
    "hitter_pr_p1_al",
    "hitter_pr_p2_al",
    "hitter_pr_p3_fx",
    "hitter_pr_p1_cs",
    "hitter_pr_p2_cs",
    "hitter_pr_p3_cs",
    "hitter_pr_p1_al_onset",
    "hitter_pr_p1_al_prec",
    "hitter_pr_p1_al_mag",
    "hitter_pr_p2_al_onset",
    "hitter_pr_p2_al_prec",
    "hitter_pr_p2_al_mag",
    "hitter_pr_p3_fx_onset",
    "hitter_pr_p3_fx_duration",
    "hitter_pr_p3_stability",

    # Joint attention
    "ja_p23_minDu",
    "ja_p23_maxDu",
    "ja_p23_avgDu",
    "ja_p23_percentage",



    # Forward swing and impact
    "receiver_ec_start_fs",
    "receiver_ec_fs_ball_racket_ratio",
    "receiver_ec_fs_ball_racket_dir",
    "receiver_ec_fs_ball_rball_dist",
    "receiver_im_racket_force",
    "receiver_im_ball_force",
    "receiver_im_rb_ang_collision",
    "receiver_im_rb_dist",
    "receiver_im_rack_wrist_dist",


    # bouncing point
    "hitter_hit_to_bouncing_point",
    "receiver_racket_to_root",
    "team_spatial_position",  # only for double




    "hitter",
    "observer",
    "pair_idx",
    "episode_label",
    "observation_label",
    "success"


]
single_features_col = [
    "id_subject",
    "skill_subject",

    # gaze related
    "pr_p1_al",
    "pr_p2_al",
    "pr_p3_fx",
    "pr_p1_sf",
    "pr_p2_sf",
    "pr_p3_sf",
    "pr_p1_al_on",
    "pr_p1_al_miDo",
    "pr_p1_al_gM",
    "pr_p2_al_on",
    "pr_p2_al_miDo",
    "pr_p2_al_gM",
    "pr_p3_fx_on",
    "pr_p3_fx_du",
    "pr_p3_stability",

    # fs related
    "ec_start_fs",
    "ec_fs_ball_racket_ratio",
    "ec_fs_ball_racket_dir",
    "ec_fs_ball_rball_dist",

    # impact related
    "im_racket_force",
    "im_ball_force",
    "im_rb_ang_collision",
    "im_rb_dist",
    "im_rack_wrist_dist",

    #bouncing point
    "bouncing_point_to_cent",


    'episode_label', 'observation_label', 'success'
]
single_features_col_num = [
    "skill_subject",


    "pr_p1_al",
    "pr_p2_al",
    "pr_p3_fx",
    "pr_p1_sf",
    "pr_p2_sf",
    "pr_p3_sf",
    "pr_p1_al_on",
    "pr_p1_al_miDo",
    "pr_p1_al_gM",
    "pr_p2_al_on",
    "pr_p2_al_miDo",
    "pr_p2_al_gM",
    "pr_p3_fx_on",
    "pr_p3_fx_du",
    "pr_p3_stability",
    "ec_start_fs",
    "ec_fs_ball_racket_ratio",
    "ec_fs_ball_racket_dir",
    "ec_fs_ball_rball_dist",
    "im_racket_force",
    "im_ball_force",
    "im_rb_ang_collision",
    "im_rb_dist",
    "im_rack_wrist_dist",

    # bouncing point
    "bouncing_point_to_cent",

    'episode_label', 'observation_label', 'success'

]
