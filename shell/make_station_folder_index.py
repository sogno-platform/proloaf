import sys
import os
import subprocess
from create_hugo_header import create_frontmatter

if __name__ == '__main__':
    input = str(sys.argv[1])
    split = input.split("/")

    path = ""

    for s in split:
        path = os.path.join(path, s)
        if os.path.exists(os.path.join(path,"_index.md")):
            continue
        with open("_index.md","w") as index_file:
            index_file.write(create_frontmatter(title=str(path), link=str(path)))
        # subprocess.call(['sh', './shell/make_station_folder_index.sh', str(path)])
