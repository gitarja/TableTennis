import glob
import subprocess

path = "F:\\users\\prasetia\\data\\TableTennis\\VideosForSync\\"

ffmpeg_path = "F:\\users\\prasetia\\projects\\Bins\\ffmpeg\\bin\\ffmpeg.exe"

for f in glob.glob(path+"*"):
    for v in glob.glob(f + "\\*.mp4"):
        a = v.replace("mp4", "mp3")
        subprocess.call(ffmpeg_path + " -i " +v + " -q:a 0 -map a " + a)

