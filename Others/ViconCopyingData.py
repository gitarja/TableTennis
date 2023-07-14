import glob
import json
import numpy as np
import os
import shutil
sub_folder = glob.glob("D:\\Backup\\Vicon\\BP-2023-04-28\\PingPong\\*")
results_path = "F:\\users\\prasetia\\data\\TableTennis\\Experiment_1_cooperation\\"


def createDir(dir):
    if os.path.isdir(dir) == False:
        os.mkdir(dir)

for sf in sub_folder:
    for ss in glob.glob(sf+"\\*"):

        if ("afternoon" in ss.lower()) | ("morning" in ss.lower()):
            date = sf.split("\\")[-1]
            session = "_M" if "morning" in ss.split("\\")[-1].lower() else "_A"
            folder_dir = results_path + date+session
            createDir(folder_dir)
            csv_files = ss + "\\*.csv"
            for f in glob.glob(csv_files):
                filename = date + session + "_" + f.split("\\")[-1]
                shutil.copyfile(f, folder_dir +"\\" +   filename) # copy csv
                shutil.copyfile(f.replace(".csv", ".c3d"), folder_dir + "\\" + filename.replace(".csv", ".c3d")) # copy c3d