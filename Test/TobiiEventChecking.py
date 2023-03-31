import glob
import gzip
import json
import numpy as np

sub_folder = glob.glob("D:\\Backup\\Tobii\\2\\*")


def checkEvents(path):
    event_file = path + "\\eventdata.gz"
    f = gzip.open(event_file, 'rb')
    file_content = f.read().decode("utf-8").split("\n")
    sync_in = []
    for c in file_content:
        if "syncport" in c:
            json_c = json.loads(c)
            if (json_c["data"]["direction"] == "in") and json_c["data"]["value"] == 0:
                sync_in.append(json_c["timestamp"])
    sync_in = np.array(sync_in)
    diff_sync = np.diff(sync_in)
    num_events = np.sum((diff_sync >= 0.19867) & (diff_sync <=0.2008))

    participant_file = path + "\\Meta\\participant"
    fp = open(participant_file, "r")
    if num_events != 2:
        print("")
        # print(path + " " + fp.read())
    else:
        idx = np.argwhere((diff_sync >= 0.19867) & (diff_sync <=0.2008))
        print(path + " " + fp.read())
        print(sync_in[idx].flatten())

for sh in sub_folder:

    for f in glob.glob(sh+"\\*"):
        if ("morning" in f.lower()) | ("afternoon" in f.lower()):
            for sf in glob.glob(f + "\\*"):
                if "aborted" in sf.lower():
                    continue
                checkEvents(sf)
        else:
            if "aborted" in f.lower():
                continue
            checkEvents(f)
