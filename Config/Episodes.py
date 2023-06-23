class EpisodesParamsDouble:
    TH_CONFIDENCE = 0.3  # threshold for valleys detection
    TH_D_RACKET = 200  # threshold for distance
    TH_WITHIN_RACKET = 50  # threshold for groupping valleys index
    TH_WITHIN = 11  # threshold for groupping valleys index
    TH_D_WALL = 250  # threshold for wall
    TH_D_TABLE = 100  # threshold for table
    TH_RACKET_SANITY = 230  # threshold for two consecutive racket sanity
    TH_SUCCESS_EPISODES = 250  # threshold for success episode
    TH_FAILURE_MID_EPISODES = 400  # threshold for the duration between success and failure episode
    TH_FAILURE_SANITY = 100 # threshold for two consecutive failure
    TH_FAILURE_EXTRAPOLATE = 230  # threshold for two consecutive failure


    def __init__(self, type="clean_ball"):
        if type == "clean_ball":
            self.TH_CONFIDENCE = 0.3  # threshold for valleys detection
            self.TH_D_RACKET = 110  # threshold for distance
            self.TH_WITHIN_RACKET = 30  # threshold for groupping valleys index
            self.TH_WITHIN = 11  # threshold for groupping valleys index
            self.TH_D_WALL = 80  # threshold for wall
            self.TH_D_TABLE = 80  # threshold for table
            self.TH_RACKET_SANITY = 230  # threshold for episode sanity
            self.TH_SUCCESS_EPISODES = 250  # threshold for success episode
            self.TH_FAILURE_MID_EPISODES = 250  # threshold for the duration between success and failure episode
            self.TH_FAILURE_SANITY = 100 # threshold for two consecutive failure


class EpisodesParamsSingle:
    TH_CONFIDENCE = 0.4  # threshold for valleys detection
    TH_D_RACKET = 200  # threshold for distance
    TH_WITHIN_RACKET = 30  # threshold for groupping valleys index
    TH_WITHIN = 11  # threshold for groupping valleys index
    TH_D_WALL = 400  # threshold for wall
    TH_D_TABLE = 200  # threshold for table
    TH_RACKET_SANITY = 230  # threshold for episode sanity
    TH_SUCCESS_EPISODES = 250  # threshold for success episode
    TH_FAILURE_MID_EPISODES = 400  # threshold for the duration between success and failure episode
    TH_FAILURE_SANITY = 100 # threshold for two consecutive failure
    TH_FAILURE_EXTRAPOLATE = 200  # threshold for two consecutive failure


    def __init__(self, type="clean_ball"):
        if type == "clean_ball":
            self.TH_CONFIDENCE = 0.4  # threshold for valleys detection
            self.TH_D_RACKET = 150  # threshold for distance
            self.TH_WITHIN_RACKET = 30  # threshold for groupping valleys index
            self.TH_WITHIN = 11  # threshold for groupping valleys index
            self.TH_D_WALL = 250  # threshold for wall
            self.TH_D_TABLE = 150  # threshold for table
            self.TH_RACKET_SANITY = 230  # threshold for episode sanity
            self.TH_SUCCESS_EPISODES = 150  # threshold for success episode
            self.TH_FAILURE_MID_EPISODES = 150  # threshold for the duration between success and failure episode
            self.TH_FAILURE_SANITY = 100 # threshold for two consecutive failure
