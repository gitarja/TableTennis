from Utils.DataReader import SubjectObjectReader
import pickle
result_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\cleaned\\2022-11-15_M\\"

file_name = "2022-11-15_M_T01"

# clean data
reader = SubjectObjectReader()
obj, sub  = reader.extractData(result_path+ file_name+ ".pkl")

# complete data

reader_comp = SubjectObjectReader()
obj_comp, sub_comp, ball_comp, tobii_comp  = reader_comp.extractData(result_path+ file_name+"_complete.pkl")

print(len(sub[0]["trajectories"].values) == len(tobii_comp[0]["trajectories"].values))

data = [obj, sub, ball_comp, tobii_comp]

with open(
        result_path  + file_name + "_complete_final.pkl",
        'wb') as f:
    pickle.dump(data, f)