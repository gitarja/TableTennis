import glob
import gzip
import json
import numpy as np

sub_folder = glob.glob("D:\\Backup\\Tobii\\2\\2023.03.15")


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
    if num_events > 0:
        idx = np.argwhere((diff_sync >= 0.19867) & (diff_sync <=0.2008))
        date_str = path.split("\\")[4]
        # print(path.split("\\")[5])
        session_str = "M" if (path.split("\\")[5].lower() == "morning") else "A"
        triad_str = fp.read().replace('{"name":"', "").replace('"}', "")
        ttl_signal = sync_in[idx].flatten()
        print(date_str + "," + session_str + "," + triad_str + ",C,")
        print(ttl_signal)
        # print(path + " " + fp.read())
        # print(date_str + "," + session_str + ","+ triad_str + ",B," + str(ttl_signal[0]) + "," + str(ttl_signal[1]))

for sh in sub_folder:

    for f in glob.glob(sh+"\\*"):
        if ("morning" in f.lower()) | ("afternoon" in f.lower()):
            for sf in glob.glob(f + "\\*"):
                if ("aborted" in f.lower()) or ("-fail" in sf.lower()):
                    continue

                checkEvents(sf)
        else:
            if ("aborted" in f.lower()):
                continue

            checkEvents(f)
