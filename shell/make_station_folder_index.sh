#! /bin/bash

if test ! -f "./repo/$1/_index.md"; then
  touch ./repo/$1/_index.md
  python ./shell/create_hugo_header.py --title=$1 --link=$1 >> ./repo/$1/_index.md
  echo "Results for $1" >> ./repo/$1/_index.md
fi
