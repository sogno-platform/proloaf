import sys
import os
import subprocess

input = str(sys.argv[1])
split = input.split("/")

path = ""

for i in range(0, len(split)):
    path = os.path.join(path, split[i])
    subprocess.call(['sh', './shell/make_station_folder_index.sh', str(path)])
