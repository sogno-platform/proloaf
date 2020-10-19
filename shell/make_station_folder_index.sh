#! /bin/bash

if test ! -f "./repo/$STATION/_index.md"; then
  echo "$STATION"
  touch ./repo/$STATION/_index.md
  python ./shell/create_hugo_header.py --title=$STATION --link=$STATION >> ./repo/$STATION/_index.md
  echo "Results for $STATION" >> ./repo/$STATION/_index.md
fi
