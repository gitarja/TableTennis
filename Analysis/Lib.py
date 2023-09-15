import numpy as np
def transformScore(df):
    df[["norm_score",
                "skill",
                "task_score",
                "max_seq",
                "avg_seq"]] = df[["norm_score",
                                                              "skill",
                                                              "task_score",
                                                              "max_seq",
                                                              "avg_seq"]].apply(np.log)


    return df