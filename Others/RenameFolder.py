import pathlib
import os
path = "D:\\Backup\\Tobii\\2\\"
files = list(pathlib.Path(path).glob('*'))

for f in files:
    f_name = f.name.split("-")[0].split(".")
    if len(f_name[-1]) == 4:
        m_a = "-" + f.name.split("-")[-1] if len(f.name.split("-")) > 1 else ""
        new_file = path + f_name[-1] + "." + f_name[1] + "." + f_name[0] + m_a
        if not os.path.exists(new_file):
            f.rename(new_file)