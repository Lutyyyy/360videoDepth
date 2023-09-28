import glob
import subprocess
import sys
import os

sys.path.append(os.getcwd())
from configs import dataset_root

mkvs = glob.glob(f"{dataset_root}/*.mkv")
for mkv in mkvs:
    scene = mkv.split(".")[0]
    subprocess.call(f"mkdir {scene}", shell=True)
    subprocess.call(f"ffmpeg -i {scene}.mkv -q:v 1 {scene}/%05d.png", shell=True)
    subprocess.call(f"rm {scene}.mkv", shell=True)
